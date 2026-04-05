"""
Multi-Agent Wind Farm Environment using PettingZoo ParallelEnv.

Each turbine is an independent agent that controls its own yaw angle.
Agents share a global reward (cooperative) — total farm power.
This enables CTDE (Centralized Training, Decentralized Execution).
"""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from floris import FlorisModel


class WindFarmMAEnv(ParallelEnv):
    """
    Multi-agent wind farm environment.

    Each turbine is an agent controlling its own yaw angle.
    All agents share the cooperative reward (farm power).
    """

    metadata = {"render_modes": ["human"], "name": "wind_farm_ma_v0"}

    def __init__(
        self,
        n_rows: int = 3,
        n_cols: int = 3,
        spacing_D: float = 5.0,
        rotor_diameter: float = 126.0,
        wind_speed_range: tuple = (5.0, 15.0),
        wind_dir_range: tuple = (250.0, 290.0),
        ti_range: tuple = (0.04, 0.10),
        max_yaw: float = 30.0,
        max_yaw_change: float = 5.0,
        absolute_yaw: bool = False,
        episode_length: int = 200,
        enable_fatigue: bool = False,
        fatigue_alpha1: float = 1e-6,
        fatigue_alpha2: float = 1e-6,
        woehler_exp: float = 4.0,
        fatigue_limit: float = 1.0,
        wind_variability: float = 0.5,
        dir_variability: float = 5.0,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_turbines = n_rows * n_cols
        self.spacing_D = spacing_D
        self.rotor_diameter = rotor_diameter
        self.max_yaw = max_yaw
        self.max_yaw_change = max_yaw_change
        self.absolute_yaw = absolute_yaw
        self.episode_length = episode_length
        self.enable_fatigue = enable_fatigue
        self.fatigue_alpha1 = fatigue_alpha1
        self.fatigue_alpha2 = fatigue_alpha2
        self.woehler_exp = woehler_exp
        self.fatigue_limit = fatigue_limit
        self.wind_speed_range = wind_speed_range
        self.wind_dir_range = wind_dir_range
        self.ti_range = ti_range
        self.wind_variability = wind_variability
        self.dir_variability = dir_variability

        # Agent IDs
        self.possible_agents = [f"turbine_{i}" for i in range(self.n_turbines)]
        self.agent_name_to_idx = {name: i for i, name in enumerate(self.possible_agents)}

        # Build layout
        spacing = spacing_D * rotor_diameter
        self.layout_x = []
        self.layout_y = []
        for i in range(n_cols):
            for j in range(n_rows):
                self.layout_x.append(i * spacing)
                self.layout_y.append(j * spacing)

        # Normalized positions for observations
        max_x = max(self.layout_x) if max(self.layout_x) > 0 else 1.0
        max_y = max(self.layout_y) if max(self.layout_y) > 0 else 1.0
        self.norm_x = [x / max_x for x in self.layout_x]
        self.norm_y = [y / max_y for y in self.layout_y]

        # Initialize FLORIS
        self.fm = FlorisModel(FlorisModel.get_defaults())
        self.fm.set(
            layout_x=self.layout_x,
            layout_y=self.layout_y,
            wind_speeds=[12.0],
            wind_directions=[270.0],
            turbulence_intensities=[0.06],
        )
        self.fm.run()
        self.rated_farm_power = self.fm.get_turbine_powers().sum()

        # RNG
        self._rng = np.random.default_rng()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Per-agent obs: [wind_speed, wind_dir, TI, my_position_x, my_position_y,
        #                  my_yaw, my_power, all_yaws(9), all_powers(9)]
        # If fatigue: + [my_del_cum, my_remaining_life]
        obs_dim = 7 + self.n_turbines * 2  # local + global yaws + global powers
        if self.enable_fatigue:
            obs_dim += 2  # my_del_cum, my_remaining_life
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Each agent controls its own yaw change
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.current_step = 0

        # Sample wind conditions
        self.wind_speed = self._rng.uniform(*self.wind_speed_range)
        self.wind_direction = self._rng.uniform(*self.wind_dir_range)
        self.turbulence_intensity = self._rng.uniform(*self.ti_range)

        # Reset state
        self.yaw_angles = np.zeros(self.n_turbines)
        self.turbine_powers = np.zeros(self.n_turbines)
        self.cumulative_del = np.zeros(self.n_turbines)
        self.remaining_life = np.ones(self.n_turbines)

        # Run initial simulation
        self._run_floris()

        observations = {agent: self._get_agent_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info() for agent in self.agents}

        return observations, infos

    def step(self, actions):
        self.current_step += 1

        # Apply actions from all agents
        for agent, action in actions.items():
            idx = self.agent_name_to_idx[agent]
            if self.absolute_yaw:
                # Action directly sets yaw misalignment relative to wind direction
                # action[0] ∈ [-1, 1]  →  yaw ∈ [-max_yaw, max_yaw] degrees
                self.yaw_angles[idx] = float(action[0]) * self.max_yaw
            else:
                # Delta-yaw: action is an incremental change per step
                delta_yaw = float(action[0]) * self.max_yaw_change
                self.yaw_angles[idx] = np.clip(
                    self.yaw_angles[idx] + delta_yaw, -self.max_yaw, self.max_yaw
                )

        # Vary wind conditions
        self.wind_speed += self._rng.normal(0, self.wind_variability * 0.1)
        self.wind_speed = np.clip(self.wind_speed, *self.wind_speed_range)
        self.wind_direction += self._rng.normal(0, self.dir_variability * 0.1)
        self.wind_direction = np.clip(self.wind_direction, *self.wind_dir_range)

        # Run FLORIS
        self._run_floris()

        # Cooperative reward: normalized farm power (same for all agents)
        farm_power = self.turbine_powers.sum()
        shared_reward = farm_power / self.rated_farm_power

        # Fatigue tracking
        if self.enable_fatigue:
            del_step = self._compute_del()
            self.cumulative_del += del_step
            self.remaining_life = np.clip(
                1.0 - self.cumulative_del / self.fatigue_limit, 0.0, 1.0
            )

        # Check termination
        terminated = self.current_step >= self.episode_length
        if terminated:
            self.agents = []

        observations = {agent: self._get_agent_obs(agent) for agent in self.possible_agents}
        rewards = {agent: float(shared_reward) for agent in self.possible_agents}
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: self._get_info() for agent in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def _run_floris(self):
        self.fm.set(
            layout_x=self.layout_x,
            layout_y=self.layout_y,
            wind_speeds=[self.wind_speed],
            wind_directions=[self.wind_direction],
            turbulence_intensities=[self.turbulence_intensity],
            yaw_angles=self.yaw_angles.reshape(1, -1),
        )
        self.fm.run()
        self.turbine_powers = self.fm.get_turbine_powers().flatten()

    def _compute_del(self) -> np.ndarray:
        m = self.woehler_exp
        m_yaw = np.abs(np.radians(self.yaw_angles)) * self.turbine_powers / 1e6
        m_flap = (self.wind_speed ** 2) * self.turbulence_intensity * np.ones(self.n_turbines)
        return self.fatigue_alpha1 * np.power(m_yaw, m) + \
               self.fatigue_alpha2 * np.power(m_flap, m)

    def _get_agent_obs(self, agent) -> np.ndarray:
        idx = self.agent_name_to_idx[agent]

        # Normalize wind params
        ws_norm = (self.wind_speed - self.wind_speed_range[0]) / \
                  (self.wind_speed_range[1] - self.wind_speed_range[0])
        wd_norm = (self.wind_direction - self.wind_dir_range[0]) / \
                  (self.wind_dir_range[1] - self.wind_dir_range[0])
        ti_norm = (self.turbulence_intensity - self.ti_range[0]) / \
                  (self.ti_range[1] - self.ti_range[0])

        # Local features
        my_yaw_norm = self.yaw_angles[idx] / self.max_yaw
        my_power_norm = self.turbine_powers[idx] / (self.rated_farm_power / self.n_turbines)

        # Global features
        all_yaws_norm = self.yaw_angles / self.max_yaw
        all_powers_norm = self.turbine_powers / (self.rated_farm_power / self.n_turbines)

        obs = np.concatenate([
            [ws_norm, wd_norm, ti_norm],
            [self.norm_x[idx], self.norm_y[idx]],
            [my_yaw_norm, my_power_norm],
            all_yaws_norm,
            all_powers_norm,
        ])

        if self.enable_fatigue:
            obs = np.concatenate([
                obs,
                [self.cumulative_del[idx] / self.fatigue_limit],
                [self.remaining_life[idx]],
            ])

        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "farm_power_mw": self.turbine_powers.sum() / 1e6,
            "turbine_powers_kw": self.turbine_powers / 1e3,
            "yaw_angles": self.yaw_angles.copy(),
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "step": self.current_step,
        }
