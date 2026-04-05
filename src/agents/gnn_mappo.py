"""
GNN-based MAPPO agent for wind farm yaw control.

Architecture (CTDE):
  GNNActorNetwork   — decentralized: per-node embedding → per-turbine action
  GNNCriticNetwork  — centralized:   mean-pooled embedding → farm-level value

Both use separate GNNEncoder instances (Structure2Vec, T=3 layers).

GNNMAPPOAgent API mirrors MAPPOAgent for drop-in compatibility with the
training loop, except inputs are graph tuples (X, A, E) rather than flat obs.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from pathlib import Path

from src.models.gnn import GNNEncoder, NODE_DIM, EDGE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Actor
# ─────────────────────────────────────────────────────────────────────────────

class GNNActorNetwork(nn.Module):
    """
    Decentralized actor: graph → per-node action distribution.

    Uses a shared GNNEncoder (same weights for all turbines).
    Action head is a small MLP applied independently to each node embedding.
    Output is Tanh-squashed to [-1, 1].
    """

    def __init__(
        self,
        act_dim:   int = 1,
        embed_dim: int = 64,
        n_layers:  int = 3,
        head_hidden: list[int] = None,
    ):
        super().__init__()
        if head_hidden is None:
            head_hidden = [64, 32]

        self.gnn = GNNEncoder(
            node_dim=NODE_DIM, edge_dim=EDGE_DIM,
            embed_dim=embed_dim, n_layers=n_layers,
        )

        # Action mean head — applied to each node's embedding independently
        layers = []
        in_dim = embed_dim
        for h in head_hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.head = nn.Sequential(*layers)

        # Learnable log-std, shared across all agents and time steps
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, X, A, E):
        """
        Returns (mean, std).

        Shapes depend on whether input has batch dim:
          X: (B, N, d) or (N, d) → mean: (B, N, act) or (N, act)
        """
        embeddings = self.gnn(X, A, E)          # (…, N, p)
        mean = torch.tanh(self.head(embeddings)) # (…, N, act_dim)
        std  = torch.exp(self.log_std.clamp(-4, 2))
        return mean, std

    @torch.no_grad()
    def get_action(self, X, A, E, deterministic: bool = False):
        """
        Sample actions for all N turbines.

        Inputs may be unbatched (N, …).
        Returns:
          actions:   (N, act_dim) numpy
          log_probs: (N,)         numpy
        """
        mean, std = self.forward(X, A, E)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)  # (N,) or (B, N)
        return action, log_prob

    def evaluate_actions(self, X, A, E, actions):
        """
        Compute log_prob and entropy for stored (obs, action) pairs.

        X: (B, N, d_node), actions: (B, N, act_dim)
        Returns log_prob: (B, N), entropy: (B, N)
        """
        mean, std = self.forward(X, A, E)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)   # (B, N)
        entropy  = dist.entropy().sum(dim=-1)           # (B, N)
        return log_prob, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Critic
# ─────────────────────────────────────────────────────────────────────────────

class GNNCriticNetwork(nn.Module):
    """
    Centralized critic: graph → scalar farm value.

    Uses a separate GNNEncoder, then mean-pools all node embeddings into
    a global farm representation, followed by an MLP head.

    Mean pooling is permutation-invariant — consistent with the
    graph-level Q̃ formulation in Dai et al. (2017).
    """

    def __init__(
        self,
        embed_dim:   int = 64,
        n_layers:    int = 3,
        head_hidden: list[int] = None,
    ):
        super().__init__()
        if head_hidden is None:
            head_hidden = [128, 64]

        self.gnn = GNNEncoder(
            node_dim=NODE_DIM, edge_dim=EDGE_DIM,
            embed_dim=embed_dim, n_layers=n_layers,
        )

        layers = []
        in_dim = embed_dim
        for h in head_hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

        nn.init.orthogonal_(self.head[-1].weight, gain=1.0)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, X, A, E):
        """
        X: (B, N, d_node) → value: (B,)
        """
        embeddings  = self.gnn(X, A, E)              # (B, N, p)
        global_emb  = embeddings.mean(dim=-2)         # (B, p)  mean pooling
        return self.head(global_emb).squeeze(-1)      # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class GNNRolloutBuffer:
    """
    Stores graph-structured trajectories for one rollout.

    Shapes (T = rollout length, N = n_turbines):
      node_feat   : (T, N, NODE_DIM)
      adj         : (T, N, N)
      edge_feat   : (T, N, N, EDGE_DIM)
      actions     : (T, N, act_dim)
      log_probs   : (T, N)
      rewards     : (T, N)
      dones       : (T,)
      values      : (T,)
    """

    def __init__(self):
        self.node_feat:  list = []
        self.adj:        list = []
        self.edge_feat:  list = []
        self.actions:    list = []
        self.log_probs:  list = []
        self.rewards:    list = []
        self.dones:      list = []
        self.values:     list = []

    def add(
        self,
        X:         np.ndarray,   # (N, NODE_DIM)
        A:         np.ndarray,   # (N, N)
        E:         np.ndarray,   # (N, N, EDGE_DIM)
        actions:   np.ndarray,   # (N, act_dim)
        log_probs: np.ndarray,   # (N,)
        rewards:   np.ndarray,   # (N,)  cooperative reward (same for all agents)
        value:     float,
        done:      bool,
    ):
        self.node_feat.append(X.copy())
        self.adj.append(A.copy())
        self.edge_feat.append(E.copy())
        self.actions.append(actions.copy())
        self.log_probs.append(log_probs.copy())
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.node_feat)

    def to_tensors(self, device: torch.device):
        def t(arr): return torch.tensor(np.stack(arr), dtype=torch.float32, device=device)
        return (
            t(self.node_feat),   # (T, N, NODE_DIM)
            t(self.adj),         # (T, N, N)
            t(self.edge_feat),   # (T, N, N, EDGE_DIM)
            t(self.actions),     # (T, N, act_dim)
            t(self.log_probs),   # (T, N)
            t(self.rewards),     # (T, N)
            torch.tensor(self.dones,  dtype=torch.float32, device=device),  # (T,)
            torch.tensor(self.values, dtype=torch.float32, device=device),  # (T,)
        )


# ─────────────────────────────────────────────────────────────────────────────
# GNN MAPPO Agent
# ─────────────────────────────────────────────────────────────────────────────

class GNNMAPPOAgent:
    """
    MAPPO agent with GNN Actor + GNN Critic (CTDE).

    Usage::

        agent = GNNMAPPOAgent(n_agents=9, act_dim=1)

        # ---- rollout ----
        X, A, E = graph_builder.build(...)          # numpy arrays
        actions, log_probs = agent.select_actions(X, A, E)
        value = agent.get_value(X, A, E)

        buffer.add(X, A, E, actions, log_probs, rewards, value, done)

        # ---- training ----
        losses = agent.update(buffer)

        # ---- persistence ----
        agent.save("results/gnn_mappo/")
        agent.load("results/gnn_mappo/")
    """

    def __init__(
        self,
        n_agents:      int   = 9,
        act_dim:       int   = 1,
        embed_dim:     int   = 64,
        gnn_layers:    int   = 3,
        lr_actor:      float = 3e-4,
        lr_critic:     float = 3e-4,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_eps:      float = 0.2,
        value_coef:    float = 0.5,
        entropy_coef:  float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs:      int   = 10,
        mini_batch_size: int = 128,
        device:        str  = "cpu",
    ):
        self.n_agents  = n_agents
        self.act_dim   = act_dim
        self.gamma     = gamma
        self.lam       = gae_lambda
        self.eps       = clip_eps
        self.vc        = value_coef
        self.ec        = entropy_coef
        self.grad_clip = max_grad_norm
        self.n_epochs  = n_epochs
        self.mb_size   = mini_batch_size
        self.device    = torch.device(device)

        self.actor  = GNNActorNetwork(act_dim=act_dim, embed_dim=embed_dim, n_layers=gnn_layers).to(self.device)
        self.critic = GNNCriticNetwork(embed_dim=embed_dim, n_layers=gnn_layers).to(self.device)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    # ------------------------------------------------------------------
    # Helpers: numpy → device tensor (no batch dim → (1, N, …))
    # ------------------------------------------------------------------

    def _to_device(self, X_np, A_np, E_np):
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        A = torch.tensor(A_np, dtype=torch.float32, device=self.device)
        E = torch.tensor(E_np, dtype=torch.float32, device=self.device)
        return X, A, E

    # ------------------------------------------------------------------
    # Rollout API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(self, X_np, A_np, E_np, deterministic: bool = False):
        """
        Select actions for all N turbines.

        Returns
        -------
        actions   : (N, act_dim) numpy
        log_probs : (N,)         numpy
        """
        X, A, E = self._to_device(X_np, A_np, E_np)
        actions, log_probs = self.actor.get_action(X, A, E, deterministic=deterministic)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    @torch.no_grad()
    def get_value(self, X_np, A_np, E_np) -> float:
        """Centralized value estimate for a single farm state."""
        X, A, E = self._to_device(X_np, A_np, E_np)
        # Add batch dim for critic
        val = self.critic(X.unsqueeze(0), A.unsqueeze(0), E.unsqueeze(0))
        return val.item()

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        Generalised Advantage Estimation.

        rewards : (T, N)  — cooperative reward (same for all agents at each t)
        values  : (T,)    — centralised value estimate
        dones   : (T,)
        Returns
        -------
        advantages : (T, N)
        returns    : (T,)  — for critic regression
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns    = torch.zeros(T, device=self.device)
        gae        = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1].item()
            mask     = 1.0 - dones[t].item()
            delta    = rewards[t] + self.gamma * next_val * mask - values[t].item()
            gae      = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t]    = gae.mean() + values[t]

        return advantages, returns

    # ------------------------------------------------------------------
    # Actor serialisation (for passing weights to worker processes)
    # ------------------------------------------------------------------

    def serialize_actor(self) -> bytes:
        """Return actor state_dict as bytes (picklable, for multiprocessing)."""
        import io
        buf = io.BytesIO()
        torch.save(self.actor.state_dict(), buf)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Internal: shared PPO mini-batch loop
    # ------------------------------------------------------------------

    def _ppo_update(self, X_all, A_all, E_all, actions, old_lp, advantages, returns):
        """Run n_epochs of mini-batch PPO on pre-computed (adv, ret) tensors."""
        total_T = X_all.shape[0]
        indices = np.arange(total_T)
        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_T, self.mb_size):
                idx = indices[start: start + self.mb_size]

                new_lp, entropy = self.actor.evaluate_actions(
                    X_all[idx], A_all[idx], E_all[idx], actions[idx]
                )
                ratio  = torch.exp(new_lp - old_lp[idx])
                surr1  = ratio * advantages[idx]
                surr2  = ratio.clamp(1 - self.eps, 1 + self.eps) * advantages[idx]
                a_loss = -torch.min(surr1, surr2).mean()
                e_loss = -entropy.mean()

                v_pred  = self.critic(X_all[idx], A_all[idx], E_all[idx])
                c_loss  = nn.functional.mse_loss(v_pred, returns[idx])

                loss = a_loss + self.vc * c_loss + self.ec * e_loss
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  self.grad_clip)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.actor_opt.step()
                self.critic_opt.step()

                actor_losses.append(a_loss.item())
                critic_losses.append(c_loss.item())
                entropies.append(-e_loss.item())

        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy":     float(np.mean(entropies)),
        }

    # ------------------------------------------------------------------
    # PPO update — single buffer (original API, unchanged)
    # ------------------------------------------------------------------

    def update(self, buffer: GNNRolloutBuffer, last_value: float = 0.0) -> dict:
        """
        Run PPO update on the stored rollout.

        Returns dict of scalar loss components for logging.
        """
        X_all, A_all, E_all, actions, old_lp, rewards, dones, values = \
            buffer.to_tensors(self.device)

        T, N = rewards.shape

        # Compute advantages
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        # Normalise advantages across all (T, N) entries
        adv_flat = advantages.reshape(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        return self._ppo_update(X_all, A_all, E_all, actions, old_lp, advantages, returns)

    # ------------------------------------------------------------------
    # PPO update — parallel trajectories (new API)
    # ------------------------------------------------------------------

    def update_from_trajs(self, trajs: list) -> dict:
        """
        Update using a list of trajectory dicts returned by parallel workers.

        Each trajectory dict contains numpy arrays:
          node_feat (T, N, d), adj (T, N, N), edge_feat (T, N, N, d),
          actions (T, N, act), log_probs (T, N), rewards (T, N), dones (T,),
          last_X (N, d), last_A (N, N), last_E (N, N, d), last_done bool.

        Values and GAE are computed here (critic stays in main process only).
        All trajectories are concatenated before the PPO update.
        """
        all_X, all_A, all_E = [], [], []
        all_acts, all_lp    = [], []
        all_adv, all_ret    = [], []

        for traj in trajs:
            X = torch.tensor(traj["node_feat"], dtype=torch.float32, device=self.device)
            A = torch.tensor(traj["adj"],        dtype=torch.float32, device=self.device)
            E = torch.tensor(traj["edge_feat"],  dtype=torch.float32, device=self.device)
            rewards = torch.tensor(traj["rewards"], dtype=torch.float32, device=self.device)
            dones   = torch.tensor(traj["dones"],   dtype=torch.float32, device=self.device)

            with torch.no_grad():
                values = self.critic(X, A, E)   # (T,)

                if traj["last_done"]:
                    last_val = 0.0
                else:
                    Xl = torch.tensor(traj["last_X"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    Al = torch.tensor(traj["last_A"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    El = torch.tensor(traj["last_E"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    last_val = self.critic(Xl, Al, El).item()

            adv, ret = self._compute_gae(rewards, values, dones, last_val)
            all_X.append(X);    all_A.append(A);    all_E.append(E)
            all_acts.append(torch.tensor(traj["actions"],   dtype=torch.float32, device=self.device))
            all_lp.append(  torch.tensor(traj["log_probs"], dtype=torch.float32, device=self.device))
            all_adv.append(adv); all_ret.append(ret)

        X_cat   = torch.cat(all_X)     # (ΣT, N, d)
        A_cat   = torch.cat(all_A)
        E_cat   = torch.cat(all_E)
        acts    = torch.cat(all_acts)
        old_lp  = torch.cat(all_lp)
        adv_cat = torch.cat(all_adv)
        ret_cat = torch.cat(all_ret)

        # Normalise advantages over all collected data
        adv_cat = (adv_cat - adv_cat.mean()) / (adv_cat.std() + 1e-8)

        return self._ppo_update(X_cat, A_cat, E_cat, acts, old_lp, adv_cat, ret_cat)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(),  path / "gnn_mappo_actor.pt")
        torch.save(self.critic.state_dict(), path / "gnn_mappo_critic.pt")

    def load(self, directory: str):
        path = Path(directory)
        self.actor.load_state_dict(
            torch.load(path / "gnn_mappo_actor.pt",  map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(path / "gnn_mappo_critic.pt", map_location=self.device)
        )
        self.actor.eval()
        self.critic.eval()
