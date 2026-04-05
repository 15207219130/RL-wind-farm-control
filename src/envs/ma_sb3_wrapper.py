"""
Wrapper to convert PettingZoo ParallelEnv to SB3-compatible environments.

Uses parameter sharing: a single PPO policy is shared across all agents.
This is a simple but effective approach for homogeneous multi-agent systems
like wind farms where all turbines share the same observation/action structure.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ParallelEnvToSB3(gym.Env):
    """
    Wraps a PettingZoo ParallelEnv for use with SB3 via parameter sharing.

    At each step, the single policy is called for each agent sequentially,
    but the environment steps all agents simultaneously (parallel step).
    """

    def __init__(self, parallel_env_fn, **env_kwargs):
        super().__init__()
        self.env = parallel_env_fn(**env_kwargs)
        self.env_kwargs = env_kwargs

        # Get spaces from first agent (all agents have same spaces)
        sample_agent = self.env.possible_agents[0]
        self.observation_space = self.env.observation_space(sample_agent)
        self.action_space = self.env.action_space(sample_agent)

        self.n_agents = len(self.env.possible_agents)
        self.agents = self.env.possible_agents

        # Expand observation to include all-agent concatenated obs
        # For parameter sharing: obs includes agent index
        agent_obs_dim = self.observation_space.shape[0]
        self.single_obs_dim = agent_obs_dim

        # Combined observation: all agents' observations stacked
        self._observations = {}
        self._current_agent_idx = 0
        self._step_actions = {}
        self._last_rewards = {}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed)
        self._observations = observations
        self._current_agent_idx = 0
        self._step_actions = {}

        # Return the first agent's observation
        first_agent = self.agents[0]
        return self._observations[first_agent], infos.get(first_agent, {})

    def step(self, action):
        """
        Collect actions from all agents, then step the environment.

        Since SB3 expects one action per step, we accumulate actions
        for all agents and step the env when all agents have acted.
        """
        current_agent = self.agents[self._current_agent_idx]
        self._step_actions[current_agent] = action
        self._current_agent_idx += 1

        if self._current_agent_idx >= self.n_agents:
            # All agents have acted — step the environment
            observations, rewards, terminations, truncations, infos = \
                self.env.step(self._step_actions)

            self._observations = observations
            self._current_agent_idx = 0
            self._step_actions = {}

            # Return mean reward (cooperative — all agents get same reward)
            mean_reward = np.mean([rewards[a] for a in self.agents])
            any_terminated = any(terminations.values())
            any_truncated = any(truncations.values())

            # Return first agent's obs for next step
            first_agent = self.agents[0]
            info = infos.get(first_agent, {})

            return self._observations[first_agent], float(mean_reward), \
                   any_terminated, any_truncated, info
        else:
            # Not all agents have acted yet — return intermediate observation
            next_agent = self.agents[self._current_agent_idx]
            return self._observations[next_agent], 0.0, False, False, {}


class SharedPolicyMAWrapper(gym.Env):
    """
    Simpler parameter-sharing wrapper: flattens multi-agent into single-agent.

    At each environment step, the policy is queried N times (once per turbine).
    All actions are collected and applied simultaneously.

    This is the recommended approach for MAPPO with parameter sharing in SB3.
    """

    def __init__(self, parallel_env_fn, **env_kwargs):
        super().__init__()
        self.env = parallel_env_fn(**env_kwargs)

        sample_agent = self.env.possible_agents[0]
        single_obs_space = self.env.observation_space(sample_agent)
        single_act_space = self.env.action_space(sample_agent)

        self.n_agents = len(self.env.possible_agents)
        self.agents = self.env.possible_agents

        # Concatenate all agents' observations and actions
        obs_dim = single_obs_space.shape[0] * self.n_agents
        act_dim = single_act_space.shape[0] * self.n_agents

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        self.single_obs_dim = single_obs_space.shape[0]
        self.single_act_dim = single_act_space.shape[0]

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed)
        obs = self._concat_obs(observations)
        info = infos.get(self.agents[0], {})
        return obs, info

    def step(self, action):
        # Split action into per-agent actions
        actions = {}
        for i, agent in enumerate(self.agents):
            start = i * self.single_act_dim
            end = start + self.single_act_dim
            actions[agent] = action[start:end]

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        obs = self._concat_obs(observations)
        reward = np.mean([rewards[a] for a in self.agents])
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        info = infos.get(self.agents[0], {})

        return obs, float(reward), terminated, truncated, info

    def _concat_obs(self, observations):
        obs_list = [observations[agent] for agent in self.agents]
        return np.concatenate(obs_list).astype(np.float32)
