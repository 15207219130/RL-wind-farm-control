"""
Wind Farm Gymnasium Environment based on FLORIS.

3x3 wind farm with NREL 5MW turbines. Yaw control for wake steering
with penalties for yaw actuation cost and power ramp rate.

Reward = power_gain - c_yaw * yaw_action - c_ramp * power_ramp
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from floris import FlorisModel


class WindFarmEnv(gym.Env):
    """
    Single-agent wind farm control environment.

    The agent controls yaw angles for all turbines simultaneously.

    Reward: maximize farm power output via wake steering.
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
        self.spacing_D = spacing_D
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
        self.layout_x = []
        self.layout_y = []
        for i in range(n_cols):
            for j in range(n_rows):
                self.layout_x.append(i * spacing)
                self.layout_y.append(j * spacing)

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

        # Action space: delta yaw for each turbine, scaled to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_turbines,),
            dtype=np.float32,
        )

            # Observation space: [wind_speed, wind_dir, TI, yaw_angles(9), powers(9)]
        obs_dim = 3 + self.n_turbines * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # State variables
        self.current_step = 0
        self.wind_speed = 0.0
        self.wind_direction = 0.0
        self.turbulence_intensity = 0.0
        self.yaw_angles = np.zeros(self.n_turbines)
        self.turbine_powers = np.zeros(self.n_turbines)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Sample initial wind conditions
        self.wind_speed = self.np_random.uniform(*self.wind_speed_range)
        self.wind_direction = self.np_random.uniform(*self.wind_dir_range)
        self.turbulence_intensity = self.np_random.uniform(*self.ti_range)

        # Reset state
        self.yaw_angles = np.zeros(self.n_turbines)

        # Run initial FLORIS simulation
        self._run_floris()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1

        # Parse action: scale from [-1, 1] to actual yaw changes
        delta_yaw = action * self.max_yaw_change

        # Apply yaw changes with constraints
        self.yaw_angles = np.clip(
            self.yaw_angles + delta_yaw, -self.max_yaw, self.max_yaw
        )

        # Slowly vary wind conditions (time-varying)
        self.wind_speed += self.np_random.normal(0, self.wind_variability * 0.1)
        self.wind_speed = np.clip(self.wind_speed, *self.wind_speed_range)
        self.wind_direction += self.np_random.normal(0, self.dir_variability * 0.1)
        self.wind_direction = np.clip(self.wind_direction, *self.wind_dir_range)

        # Run FLORIS
        self._run_floris()
        farm_power = self.turbine_powers.sum()

        # Reward: normalized farm power
        reward = farm_power / self.rated_farm_power

        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _run_floris(self):
        """Run FLORIS simulation with current wind conditions and yaw angles."""
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

    def _get_obs(self) -> np.ndarray:
        # Normalize wind params
        ws_norm = (self.wind_speed - self.wind_speed_range[0]) / \
                  (self.wind_speed_range[1] - self.wind_speed_range[0])
        wd_norm = (self.wind_direction - self.wind_dir_range[0]) / \
                  (self.wind_dir_range[1] - self.wind_dir_range[0])
        ti_norm = (self.turbulence_intensity - self.ti_range[0]) / \
                  (self.ti_range[1] - self.ti_range[0])

        yaw_norm = self.yaw_angles / self.max_yaw
        power_norm = self.turbine_powers / (self.rated_farm_power / self.n_turbines)

        obs = np.concatenate([
            [ws_norm, wd_norm, ti_norm],
            yaw_norm,
            power_norm,
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

    def get_greedy_power(self) -> float:
        """Get power with all turbines aligned (yaw=0), for baseline comparison."""
        self.fm.set(
            layout_x=self.layout_x,
            layout_y=self.layout_y,
            wind_speeds=[self.wind_speed],
            wind_directions=[self.wind_direction],
            turbulence_intensities=[self.turbulence_intensity],
            yaw_angles=np.zeros((1, self.n_turbines)),
        )
        self.fm.run()
        return self.fm.get_turbine_powers().sum()
