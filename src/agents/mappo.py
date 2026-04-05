"""
Custom MAPPO (Multi-Agent PPO) Implementation.

Architecture:
- ActorNetwork: decentralized policy (local obs → action), parameter-shared across agents
- CriticNetwork: centralized value function (global state → value), CTDE paradigm
- RolloutBuffer: stores trajectories for one rollout iteration
- MAPPOAgent: orchestrates rollout collection + network updates

Reference: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
           (MAPPO paper, NeurIPS 2022)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from pathlib import Path


# --------------------------------------------------------------------------- #
# Neural Networks
# --------------------------------------------------------------------------- #

class ActorNetwork(nn.Module):
    """
    Decentralized actor: maps per-agent local observation to action distribution.

    Outputs mean and log_std of a Gaussian; action is sampled from Normal(mean, std).
    Tanh squashing ensures actions stay in [-1, 1].
    Parameter-shared: the same weights are used for every agent.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int] = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]

        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, act_dim)
        # Log std is a learnable parameter (not input-dependent), initialized near 0
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Initialize output layer with small weights for stable early training
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor):
        """Returns (mean, std) of action distribution."""
        x = self.backbone(obs)
        mean = torch.tanh(self.mean_head(x))
        std = torch.exp(self.log_std.clamp(-4, 2))
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample action and compute log probability.

        Returns:
            action: shape (..., act_dim), clipped to [-1, 1]
            log_prob: shape (...,), sum over act_dim
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Compute log_prob and entropy for given (obs, action) pairs (used in PPO update).

        Returns:
            log_prob: shape (batch,)
            entropy: shape (batch,)
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralized critic: maps global state (all agents' observations concatenated) to value.

    Uses a larger network since it processes full farm state.
    """

    def __init__(self, global_state_dim: int, hidden: list[int] = None):
        super().__init__()
        if hidden is None:
            hidden = [512, 256, 128]

        layers = []
        in_dim = global_state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize last layer with small weights
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Returns value estimate, shape (..., 1)."""
        return self.net(global_state)


# --------------------------------------------------------------------------- #
# Rollout Buffer
# --------------------------------------------------------------------------- #

class RolloutBuffer:
    """
    Stores one complete rollout (T steps × N agents).

    Shapes:
        obs         : (T, N, obs_dim)
        actions     : (T, N, act_dim)
        log_probs   : (T, N)
        rewards     : (T, N)
        dones       : (T,)          — episode done flag (same for all agents)
        values      : (T,)          — centralized value estimate
        global_states: (T, global_state_dim)
    """

    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.global_states: list[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,           # (N, obs_dim)
        global_state: np.ndarray,  # (global_state_dim,)
        actions: np.ndarray,       # (N, act_dim)
        log_probs: np.ndarray,     # (N,)
        rewards: np.ndarray,       # (N,) or scalar (cooperative)
        value: float,
        done: bool,
    ):
        self.obs.append(obs.copy())
        self.global_states.append(global_state.copy())
        self.actions.append(actions.copy())
        self.log_probs.append(log_probs.copy())
        if np.isscalar(rewards):
            self.rewards.append(np.full(obs.shape[0], rewards, dtype=np.float32))
        else:
            self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)

    def to_tensors(self, device: torch.device):
        """Convert stored lists to stacked tensors."""
        obs = torch.tensor(np.stack(self.obs), dtype=torch.float32, device=device)
        actions = torch.tensor(np.stack(self.actions), dtype=torch.float32, device=device)
        log_probs = torch.tensor(np.stack(self.log_probs), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.stack(self.rewards), dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        global_states = torch.tensor(
            np.stack(self.global_states), dtype=torch.float32, device=device
        )
        return obs, actions, log_probs, rewards, dones, values, global_states


# --------------------------------------------------------------------------- #
# MAPPO Agent
# --------------------------------------------------------------------------- #

class MAPPOAgent:
    """
    Main MAPPO training class.

    Usage:
        agent = MAPPOAgent(obs_dim=40, act_dim=1, n_agents=9, global_state_dim=360)

        # Rollout
        actions, log_probs = agent.select_actions(obs_array)   # obs_array: (N, obs_dim)
        value = agent.get_value(global_state)                   # global_state: (global_state_dim,)

        # Training
        losses = agent.update(buffer)

        # Persistence
        agent.save("results/mappo_v2/")
        agent.load("results/mappo_v2/")
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_agents: int,
        global_state_dim: int,
        actor_hidden: list[int] = None,
        critic_hidden: list[int] = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        mini_batch_size: int = 256,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.global_state_dim = global_state_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size

        self.device = torch.device(device)

        # Networks
        self.actor = ActorNetwork(obs_dim, act_dim, actor_hidden).to(self.device)
        self.critic = CriticNetwork(global_state_dim, critic_hidden).to(self.device)

        # Separate optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray, deterministic: bool = False):
        """
        Select actions for all N agents.

        Args:
            obs: (N, obs_dim) numpy array
        Returns:
            actions: (N, act_dim) numpy
            log_probs: (N,) numpy
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions, log_probs = self.actor.get_action(obs_t, deterministic=deterministic)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        """Evaluate centralized value for a single global state."""
        gs_t = torch.tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        value = self.critic(gs_t)
        return value.item()

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: torch.Tensor,   # (T, N)
        values: torch.Tensor,    # (T,) centralized
        dones: torch.Tensor,     # (T,)
        last_value: float,
    ):
        """
        Compute GAE advantages and returns.

        We use a single centralized value stream.
        Advantages are computed agent-wise using the cooperative reward,
        but the same centralized value baseline is shared.

        Returns:
            advantages: (T, N)
            returns: (T,) discounted returns for critic
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros(T, device=self.device)

        last_val = last_value
        gae = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else values[t + 1].item()
            mask = 1.0 - dones[t].item()
            delta = rewards[t] + self.gamma * next_val * mask - values[t].item()
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t].mean() + values[t]  # returns for critic training

        return advantages, returns

    # ------------------------------------------------------------------
    # Network update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        """
        Run PPO update on stored rollout.

        Args:
            buffer: filled RolloutBuffer
            last_value: bootstrap value for last state (0 if episode ended)
        Returns:
            dict with loss components (for logging)
        """
        obs, actions, old_log_probs, rewards, dones, values, global_states = \
            buffer.to_tensors(self.device)

        # obs: (T, N, obs_dim)
        T, N, _ = obs.shape

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        # Normalize advantages (improves stability)
        adv_flat = advantages.reshape(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Flatten time and agent dims for mini-batch sampling
        obs_flat = obs.reshape(T * N, self.obs_dim)               # (T*N, obs_dim)
        actions_flat = actions.reshape(T * N, self.act_dim)       # (T*N, act_dim)
        old_lp_flat = old_log_probs.reshape(T * N)                # (T*N,)
        adv_flat = advantages.reshape(T * N)                      # (T*N,)

        # Critic: repeat global state per agent for matched indexing
        gs_repeated = global_states.unsqueeze(1).expand(T, N, -1)  # (T, N, gs_dim)
        gs_flat = gs_repeated.reshape(T * N, self.global_state_dim)  # (T*N, gs_dim)
        # Returns: same for all agents at each step
        ret_repeated = returns.unsqueeze(1).expand(T, N)           # (T, N)
        ret_flat = ret_repeated.reshape(T * N)                     # (T*N,)

        # Mini-batch PPO updates
        total_samples = T * N
        indices = np.arange(total_samples)

        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, self.mini_batch_size):
                batch_idx = indices[start: start + self.mini_batch_size]

                obs_b = obs_flat[batch_idx]
                act_b = actions_flat[batch_idx]
                old_lp_b = old_lp_flat[batch_idx]
                adv_b = adv_flat[batch_idx]
                gs_b = gs_flat[batch_idx]
                ret_b = ret_flat[batch_idx]

                # ---- Actor update ----
                new_lp, entropy = self.actor.evaluate_actions(obs_b, act_b)
                ratio = torch.exp(new_lp - old_lp_b)

                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # ---- Critic update ----
                value_pred = self.critic(gs_b).squeeze(-1)
                critic_loss = nn.functional.mse_loss(value_pred, ret_b)

                # ---- Combined loss ----
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(-entropy_loss.item())

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "mappo_v2_actor.pt")
        torch.save(self.critic.state_dict(), path / "mappo_v2_critic.pt")

    def load(self, directory: str):
        path = Path(directory)
        self.actor.load_state_dict(
            torch.load(path / "mappo_v2_actor.pt", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(path / "mappo_v2_critic.pt", map_location=self.device)
        )
        self.actor.eval()
        self.critic.eval()
