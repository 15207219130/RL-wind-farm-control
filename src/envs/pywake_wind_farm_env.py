"""
Wind Farm Gymnasium Environment based on PyWake.

Same interface as WindFarmEnv (FLORIS-based) to enable cross-simulator comparison.
Uses Bastankhah Gaussian wake model with NREL 5MW turbine definition.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from py_wake.site import UniformSite
from py_wake import BastankhahGaussian
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


def make_nrel5mw():
    """Create NREL 5MW turbine for PyWake."""
    ws = np.arange(3, 26, 1.0)
    power_kw = np.array([
        0, 40, 177, 403, 737, 1187, 1771, 2518, 3448, 4562, 5000, 5000,
        5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 0
    ]) * 1.0
    ct = np.array([
        0, 0.99, 0.99, 0.97, 0.955, 0.92, 0.855, 0.76, 0.67, 0.57, 0.45, 0.35,
        0.28, 0.22, 0.18, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0
    ])
    return WindTurbine(
        name='NREL5MW', diameter=126.0, hub_height=90.0,
        powerCtFunction=PowerCtTabular(ws, power_kw, 'kw', ct),
    )


class PyWakeWindFarmEnv(gym.Env):
    """
    PyWake-based wind farm environment. Same API as WindFarmEnv (FLORIS).

    Reward = normalized farm power (maximize energy capture).
    """

    metadata = {"render_modes": ["human"]}

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
        episode_length: int = 200,
        wind_variability: float = 0.5,
        dir_variability: float = 5.0,
    ):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_turbines = n_rows * n_cols
        self.rotor_diameter = rotor_diameter
        self.max_yaw = max_yaw
        self.max_yaw_change = max_yaw_change
        self.episode_length = episode_length
        self.wind_speed_range = wind_speed_range
        self.wind_dir_range = wind_dir_range
        self.ti_range = ti_range
        self.wind_variability = wind_variability
        self.dir_variability = dir_variability

        # Build layout
        spacing = spacing_D * rotor_diameter
        self.layout_x = np.array([i * spacing for i in range(n_cols) for _ in range(n_rows)])
        self.layout_y = np.array([j * spacing for _ in range(n_cols) for j in range(n_rows)])

        # Initialize PyWake
        self.turbine = make_nrel5mw()
        self.site = UniformSite(p_wd=[1], ti=0.06)
        self.wf_model = BastankhahGaussian(self.site, self.turbine)

        # Get rated power for normalization
        sim = self.wf_model(self.layout_x, self.layout_y, ws=12.0, wd=270.0, yaw=np.zeros(self.n_turbines))
        self.rated_farm_power = sim.Power.values.flatten().sum()

        # Spaces (same as FLORIS env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_turbines,), dtype=np.float32)
        obs_dim = 3 + self.n_turbines * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # State
        self.current_step = 0
        self.wind_speed = 0.0
        self.wind_direction = 0.0
        self.turbulence_intensity = 0.0
        self.yaw_angles = np.zeros(self.n_turbines)
        self.turbine_powers = np.zeros(self.n_turbines)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.wind_speed = self.np_random.uniform(*self.wind_speed_range)
        self.wind_direction = self.np_random.uniform(*self.wind_dir_range)
        self.turbulence_intensity = self.np_random.uniform(*self.ti_range)
        self.yaw_angles = np.zeros(self.n_turbines)
        self._run_sim()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        delta_yaw = action * self.max_yaw_change
        self.yaw_angles = np.clip(self.yaw_angles + delta_yaw, -self.max_yaw, self.max_yaw)

        self.wind_speed += self.np_random.normal(0, self.wind_variability * 0.1)
        self.wind_speed = np.clip(self.wind_speed, *self.wind_speed_range)
        self.wind_direction += self.np_random.normal(0, self.dir_variability * 0.1)
        self.wind_direction = np.clip(self.wind_direction, *self.wind_dir_range)

        self._run_sim()
        farm_power = self.turbine_powers.sum()

        reward = farm_power / self.rated_farm_power
        terminated = self.current_step >= self.episode_length
        return self._get_obs(), float(reward), terminated, False, self._get_info()

    def _run_sim(self):
        ws = np.clip(self.wind_speed, 3.5, 24.5)
        sim = self.wf_model(
            self.layout_x, self.layout_y,
            ws=ws, wd=self.wind_direction,
            yaw=self.yaw_angles,
        )
        self.turbine_powers = np.maximum(sim.Power.values.flatten(), 0.0)

    def _get_obs(self) -> np.ndarray:
        ws_norm = (self.wind_speed - self.wind_speed_range[0]) / (self.wind_speed_range[1] - self.wind_speed_range[0])
        wd_norm = (self.wind_direction - self.wind_dir_range[0]) / (self.wind_dir_range[1] - self.wind_dir_range[0])
        ti_norm = (self.turbulence_intensity - self.ti_range[0]) / (self.ti_range[1] - self.ti_range[0])
        yaw_norm = self.yaw_angles / self.max_yaw
        power_norm = self.turbine_powers / (self.rated_farm_power / self.n_turbines)
        return np.concatenate([[ws_norm, wd_norm, ti_norm], yaw_norm, power_norm]).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "farm_power_mw": self.turbine_powers.sum() / 1e6,
            "turbine_powers_kw": self.turbine_powers / 1e3,
            "yaw_angles": self.yaw_angles.copy(),
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "step": self.current_step,
        }

    def get_greedy_power(self) -> float:
        ws = np.clip(self.wind_speed, 3.5, 24.5)
        sim = self.wf_model(self.layout_x, self.layout_y, ws=ws, wd=self.wind_direction, yaw=np.zeros(self.n_turbines))
        return np.maximum(sim.Power.values.flatten(), 0.0).sum()
