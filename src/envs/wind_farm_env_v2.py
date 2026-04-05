"""
Wind Farm Environment v2 — Lifecycle-aware with realistic wind patterns.

Key improvements over v1:
1. Diurnal wind pattern (not random walk) — agent can anticipate future wind
2. Episode = turbine lifetime (ends when any turbine fatigue hits limit)
3. Reward = lifetime total energy (not instantaneous power)
4. Wind forecast horizon in observation — enables forward-looking decisions
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from floris import FlorisModel


class WindFarmEnvV2(gym.Env):
    """
    Lifecycle-aware wind farm control environment.

    Key design choices:
    - Episode represents turbine operational life (variable length)
    - Wind follows realistic diurnal + stochastic pattern
    - Agent observes wind forecast for planning ahead
    - Reward balances instantaneous power and terminal life value
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_rows: int = 3,
        n_cols: int = 3,
        spacing_D: float = 5.0,
        rotor_diameter: float = 126.0,
        max_yaw: float = 30.0,
        max_yaw_change: float = 5.0,
        # --- Wind pattern ---
        wind_speed_mean: float = 9.0,
        wind_speed_amplitude: float = 3.0,      # diurnal variation amplitude
        wind_speed_noise_std: float = 0.5,       # stochastic component
        wind_dir_mean: float = 270.0,
        wind_dir_amplitude: float = 10.0,
        wind_dir_noise_std: float = 2.0,
        ti_mean: float = 0.06,
        ti_std: float = 0.01,
        forecast_horizon: int = 12,              # how many future steps agent can see
        # --- Time ---
        steps_per_hour: float = 6.0,             # 1 step = 10 minutes
        hours_per_day: float = 24.0,
        max_days: int = 30,                      # max episode = 30 days
        # --- Fatigue ---
        fatigue_alpha1: float = 5e-4,
        fatigue_alpha2: float = 1e-5,
        woehler_exp: float = 4.0,
        fatigue_limit: float = 1.0,              # normalized: 1.0 = end of life
        initial_fatigue_range: tuple = (0.0, 0.5),  # start with some pre-existing damage
        # --- Reward design ---
        terminal_life_bonus: float = 50.0,       # bonus per unit of remaining life at episode end
    ):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_turbines = n_rows * n_cols
        self.rotor_diameter = rotor_diameter
        self.max_yaw = max_yaw
        self.max_yaw_change = max_yaw_change

        # Wind pattern parameters
        self.wind_speed_mean = wind_speed_mean
        self.wind_speed_amplitude = wind_speed_amplitude
        self.wind_speed_noise_std = wind_speed_noise_std
        self.wind_dir_mean = wind_dir_mean
        self.wind_dir_amplitude = wind_dir_amplitude
        self.wind_dir_noise_std = wind_dir_noise_std
        self.ti_mean = ti_mean
        self.ti_std = ti_std
        self.forecast_horizon = forecast_horizon

        # Time parameters
        self.steps_per_hour = steps_per_hour
        self.hours_per_day = hours_per_day
        self.steps_per_day = int(steps_per_hour * hours_per_day)
        self.max_steps = int(max_days * self.steps_per_day)

        # Fatigue parameters
        self.fatigue_alpha1 = fatigue_alpha1
        self.fatigue_alpha2 = fatigue_alpha2
        self.woehler_exp = woehler_exp
        self.fatigue_limit = fatigue_limit
        self.initial_fatigue_range = initial_fatigue_range
        self.terminal_life_bonus = terminal_life_bonus

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
            layout_x=self.layout_x, layout_y=self.layout_y,
            wind_speeds=[12.0], wind_directions=[270.0],
            turbulence_intensities=[0.06],
        )
        self.fm.run()
        self.rated_farm_power = self.fm.get_turbine_powers().sum()

        # Action space: yaw changes for each turbine
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_turbines,), dtype=np.float32
        )

        # Observation space:
        # [time_of_day(sin,cos), wind_speed, wind_dir, TI,
        #  yaw_angles(9), powers(9),
        #  cumulative_del(9), remaining_life(9),
        #  wind_speed_forecast(H), wind_dir_forecast(H)]
        obs_dim = (2          # time encoding (sin, cos)
                 + 3          # current wind (speed, dir, TI)
                 + self.n_turbines  # yaw angles
                 + self.n_turbines  # powers
                 + self.n_turbines  # cumulative DEL
                 + self.n_turbines  # remaining life
                 + forecast_horizon  # future wind speed
                 + forecast_horizon) # future wind direction
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self.current_step = 0
        self.hour_of_day = 0.0
        self.wind_speed = 0.0
        self.wind_direction = 0.0
        self.turbulence_intensity = 0.0
        self.yaw_angles = np.zeros(self.n_turbines)
        self.turbine_powers = np.zeros(self.n_turbines)
        self.cumulative_del = np.zeros(self.n_turbines)
        self.remaining_life = np.ones(self.n_turbines)
        self.total_energy = 0.0  # cumulative energy in MWh
        self._wind_noise_seq = None
        self._dir_noise_seq = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_energy = 0.0

        # Random starting hour
        self.hour_of_day = self.np_random.uniform(0, 24)

        # Pre-generate wind noise sequence for the whole episode
        # This allows computing forecasts (agent can see future noise)
        self._wind_noise_seq = self.np_random.normal(
            0, self.wind_speed_noise_std, size=self.max_steps + self.forecast_horizon
        )
        self._dir_noise_seq = self.np_random.normal(
            0, self.wind_dir_noise_std, size=self.max_steps + self.forecast_horizon
        )

        # Initialize wind from diurnal pattern
        self._update_wind()

        # Reset yaw
        self.yaw_angles = np.zeros(self.n_turbines)

        # Initialize fatigue: optionally start with some pre-existing damage
        # This creates heterogeneous scenarios (some turbines are "older")
        self.cumulative_del = self.np_random.uniform(
            *self.initial_fatigue_range, size=self.n_turbines
        )
        self.remaining_life = np.clip(
            1.0 - self.cumulative_del / self.fatigue_limit, 0.0, 1.0
        )

        self._run_floris()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1

        # Apply yaw action
        delta_yaw = action * self.max_yaw_change
        self.yaw_angles = np.clip(
            self.yaw_angles + delta_yaw, -self.max_yaw, self.max_yaw
        )

        # Advance time
        self.hour_of_day += 1.0 / self.steps_per_hour
        if self.hour_of_day >= 24.0:
            self.hour_of_day -= 24.0

        # Update wind from diurnal pattern + noise
        self._update_wind()

        # Run FLORIS
        self._run_floris()

        # --- Reward design ---
        farm_power = self.turbine_powers.sum()
        energy_mwh = farm_power / 1e6 / self.steps_per_hour  # power(W) → energy(MWh)
        self.total_energy += energy_mwh

        # Instantaneous reward = normalized power
        reward = farm_power / self.rated_farm_power

        # Fatigue update
        del_step = self._compute_del()
        self.cumulative_del += del_step
        self.remaining_life = np.clip(
            1.0 - self.cumulative_del / self.fatigue_limit, 0.0, 1.0
        )

        # --- Termination conditions ---
        # 1) Any turbine fatigue reaches limit → episode ends (turbine "dies")
        fatigue_exceeded = np.any(self.remaining_life <= 0)
        # 2) Max episode length
        time_exceeded = self.current_step >= self.max_steps

        terminated = fatigue_exceeded or time_exceeded

        # Terminal bonus: reward remaining life at end of episode
        # This incentivizes keeping turbines healthy
        if terminated:
            reward += self.terminal_life_bonus * np.mean(self.remaining_life)

        truncated = False
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _update_wind(self):
        """Compute wind from diurnal pattern + pre-generated noise."""
        t = self.current_step

        # Diurnal wind speed: peaks in afternoon (~14:00), lowest at night (~4:00)
        # v(t) = v_mean + A * sin(2π * (hour - 6) / 24)
        hour = self.hour_of_day
        diurnal_speed = self.wind_speed_mean + \
            self.wind_speed_amplitude * np.sin(2 * np.pi * (hour - 6) / 24)
        self.wind_speed = np.clip(
            diurnal_speed + self._wind_noise_seq[t], 3.0, 25.0
        )

        # Diurnal wind direction: slight rotation through the day
        diurnal_dir = self.wind_dir_mean + \
            self.wind_dir_amplitude * np.sin(2 * np.pi * (hour - 12) / 24)
        self.wind_direction = diurnal_dir + self._dir_noise_seq[t]

        # Turbulence: higher during day (convective), lower at night (stable)
        ti_diurnal = self.ti_mean + 0.02 * np.sin(2 * np.pi * (hour - 8) / 24)
        self.turbulence_intensity = np.clip(
            ti_diurnal + self.np_random.normal(0, self.ti_std), 0.02, 0.20
        )

    def _get_wind_forecast(self) -> tuple:
        """Return future wind speed and direction for forecast_horizon steps."""
        t = self.current_step
        H = self.forecast_horizon

        future_speeds = []
        future_dirs = []
        for dt in range(1, H + 1):
            future_hour = self.hour_of_day + dt / self.steps_per_hour
            if future_hour >= 24.0:
                future_hour -= 24.0

            # Diurnal component (deterministic, fully predictable)
            speed = self.wind_speed_mean + \
                self.wind_speed_amplitude * np.sin(2 * np.pi * (future_hour - 6) / 24)
            direction = self.wind_dir_mean + \
                self.wind_dir_amplitude * np.sin(2 * np.pi * (future_hour - 12) / 24)

            # Add noise (agent can see the pre-generated noise = perfect short-term forecast)
            if t + dt < len(self._wind_noise_seq):
                speed += self._wind_noise_seq[t + dt]
                direction += self._dir_noise_seq[t + dt]

            future_speeds.append(np.clip(speed, 3.0, 25.0))
            future_dirs.append(direction)

        return np.array(future_speeds), np.array(future_dirs)

    def _run_floris(self):
        self.fm.set(
            layout_x=self.layout_x, layout_y=self.layout_y,
            wind_speeds=[self.wind_speed],
            wind_directions=[self.wind_direction],
            turbulence_intensities=[self.turbulence_intensity],
            yaw_angles=self.yaw_angles.reshape(1, -1),
        )
        self.fm.run()
        self.turbine_powers = self.fm.get_turbine_powers().flatten()

    def _compute_del(self) -> np.ndarray:
        """Per-step fatigue proxy."""
        m = self.woehler_exp
        m_yaw = np.abs(np.radians(self.yaw_angles)) * self.turbine_powers / 1e6
        m_flap = (self.wind_speed ** 2) * self.turbulence_intensity * np.ones(self.n_turbines)
        return self.fatigue_alpha1 * np.power(m_yaw, m) + \
               self.fatigue_alpha2 * np.power(m_flap, m)

    def _get_obs(self) -> np.ndarray:
        # Time encoding (cyclical)
        hour_rad = 2 * np.pi * self.hour_of_day / 24.0
        time_sin = np.sin(hour_rad)
        time_cos = np.cos(hour_rad)

        # Normalize current wind
        ws_norm = (self.wind_speed - 3.0) / 22.0  # [3, 25] → [0, 1]
        wd_norm = (self.wind_direction - 250.0) / 40.0  # rough normalization
        ti_norm = (self.turbulence_intensity - 0.02) / 0.18

        # Normalize yaw and power
        yaw_norm = self.yaw_angles / self.max_yaw
        power_norm = self.turbine_powers / (self.rated_farm_power / self.n_turbines)

        # Fatigue state
        del_norm = self.cumulative_del / self.fatigue_limit

        # Wind forecast (normalized)
        future_speeds, future_dirs = self._get_wind_forecast()
        speed_forecast_norm = (future_speeds - 3.0) / 22.0
        dir_forecast_norm = (future_dirs - 250.0) / 40.0

        obs = np.concatenate([
            [time_sin, time_cos],
            [ws_norm, wd_norm, ti_norm],
            yaw_norm,
            power_norm,
            del_norm,
            self.remaining_life,
            speed_forecast_norm,
            dir_forecast_norm,
        ])
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "farm_power_mw": self.turbine_powers.sum() / 1e6,
            "turbine_powers_kw": self.turbine_powers / 1e3,
            "yaw_angles": self.yaw_angles.copy(),
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "hour_of_day": self.hour_of_day,
            "step": self.current_step,
            "total_energy_mwh": self.total_energy,
            "cumulative_del": self.cumulative_del.copy(),
            "remaining_life": self.remaining_life.copy(),
            "min_remaining_life": float(self.remaining_life.min()),
        }
