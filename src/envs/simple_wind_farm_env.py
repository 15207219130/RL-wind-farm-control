"""
Simple Wind Farm Environment with Jensen Wake Model.

No FLORIS dependency. Analytical wake model + yaw-induced wake deflection.
Reward = normalized_power - c_yaw * yaw_cost

3x3 wind farm, NREL 5MW turbines.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimpleWindFarmEnv(gym.Env):
    """
    Wind farm yaw control environment using Jensen/Park wake model.

    The agent sets absolute yaw angles to steer wakes and maximize total power,
    while paying a cost for yaw actuation (change from previous step).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_rows: int = 3,
        n_cols: int = 3,
        spacing_D: float = 5.0,
        rotor_diameter: float = 126.0,
        hub_height: float = 90.0,
        rated_power: float = 5e6,  # 5 MW per turbine
        rated_wind_speed: float = 11.4,
        cut_in: float = 3.0,
        cut_out: float = 25.0,
        wind_speed_range: tuple = (6.0, 14.0),
        wind_dir_range: tuple = (260.0, 280.0),
        max_yaw: float = 30.0,
        episode_length: int = 200,
        wind_variability: float = 0.3,
        dir_variability: float = 2.0,
        c_yaw: float = 0.02,       # yaw actuation cost coefficient
        wake_decay: float = 0.075,  # Jensen wake decay constant (onshore)
        Ct: float = 0.8,           # thrust coefficient
    ):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_turbines = n_rows * n_cols
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.rated_wind_speed = rated_wind_speed
        self.cut_in = cut_in
        self.cut_out = cut_out
        self.max_yaw = max_yaw
        self.episode_length = episode_length
        self.wind_speed_range = wind_speed_range
        self.wind_dir_range = wind_dir_range
        self.wind_variability = wind_variability
        self.dir_variability = dir_variability
        self.c_yaw = c_yaw
        self.wake_decay = wake_decay
        self.Ct = Ct

        # Build grid layout
        spacing = spacing_D * rotor_diameter
        self.layout_x = np.zeros(self.n_turbines)
        self.layout_y = np.zeros(self.n_turbines)
        self.norm_x = np.zeros(self.n_turbines)  # normalized [0,1]
        self.norm_y = np.zeros(self.n_turbines)
        max_coord = max((n_cols - 1), (n_rows - 1)) * spacing
        idx = 0
        for i in range(n_cols):
            for j in range(n_rows):
                self.layout_x[idx] = i * spacing
                self.layout_y[idx] = j * spacing
                self.norm_x[idx] = i / max(n_cols - 1, 1)
                self.norm_y[idx] = j / max(n_rows - 1, 1)
                idx += 1

        # Rated farm power (all turbines at rated, no wake)
        self.rated_farm_power = self.n_turbines * rated_power

        # Action: absolute yaw for each turbine, [-1, 1] mapped to [-max_yaw, max_yaw]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_turbines,), dtype=np.float32,
        )

        # Obs: [wind_speed, wind_dir, layout_x(9), layout_y(9), yaw_angles(9), powers(9)]
        obs_dim = 2 + self.n_turbines * 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )

        # State
        self.current_step = 0
        self.wind_speed = 10.0
        self.wind_direction = 270.0
        self.yaw_angles = np.zeros(self.n_turbines)
        self.turbine_powers = np.zeros(self.n_turbines)
        self.prev_yaw_angles = np.zeros(self.n_turbines)

    # ----- Power curve -----
    def _power_curve(self, u: float) -> float:
        """Simplified cubic power curve for a single turbine."""
        if u < self.cut_in or u > self.cut_out:
            return 0.0
        if u >= self.rated_wind_speed:
            return self.rated_power
        return self.rated_power * ((u - self.cut_in) / (self.rated_wind_speed - self.cut_in)) ** 3

    # ----- Jensen wake model with yaw deflection -----
    def _compute_powers(self) -> np.ndarray:
        """
        Compute turbine powers using Jensen model + yaw cosine loss + wake deflection.

        Key physics:
        - Yawed turbine: power *= cos(yaw)^p_p (p_p ~ 1.88, Gebraad et al.)
        - Yawed turbine: reduced thrust -> Ct_eff = Ct * cos(yaw)^2
        - Wake deflection: lateral shift ~ sin(yaw)*cos(yaw) (Jimenez)
        - Jensen deficit: (1 - sqrt(1 - Ct_eff)) * (D / (D + 2*k*x))^2
        - Multiple wakes: RSS combination
        """
        wd_rad = np.deg2rad(self.wind_direction)
        yaw_rad = np.deg2rad(self.yaw_angles)

        cos_wd, sin_wd = np.cos(wd_rad), np.sin(wd_rad)
        x_rot = self.layout_x * cos_wd + self.layout_y * sin_wd
        y_rot = -self.layout_x * sin_wd + self.layout_y * cos_wd

        order = np.argsort(x_rot)
        D = self.rotor_diameter
        k = self.wake_decay

        deficit_sq = np.zeros(self.n_turbines)

        for i_idx in range(len(order)):
            i = order[i_idx]
            for j_idx in range(i_idx + 1, len(order)):
                j = order[j_idx]
                dx = x_rot[j] - x_rot[i]
                if dx <= 0:
                    continue

                Ct_eff = self.Ct * np.cos(yaw_rad[i]) ** 2

                wake_diameter = D + 2 * k * dx
                wake_radius = wake_diameter / 2.0

                # Jimenez deflection
                deflection = 0.3 * dx * np.sin(yaw_rad[i]) * np.cos(yaw_rad[i]) * \
                             (D / wake_diameter)

                dy = abs(y_rot[j] - y_rot[i] - deflection)

                if dy < wake_radius:
                    overlap = max(0.0, 1.0 - dy / wake_radius)
                    deficit_i = overlap * (1 - np.sqrt(1 - Ct_eff)) * (D / wake_diameter) ** 2
                    deficit_sq[j] += deficit_i ** 2

        total_deficit = np.sqrt(deficit_sq)
        total_deficit = np.clip(total_deficit, 0, 0.95)
        u_eff = self.wind_speed * (1 - total_deficit)

        powers = np.zeros(self.n_turbines)
        p_p = 1.88
        for i in range(self.n_turbines):
            base_power = self._power_curve(u_eff[i])
            yaw_loss = np.cos(yaw_rad[i]) ** p_p
            powers[i] = base_power * yaw_loss

        return powers

    # ----- Gym interface -----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.wind_speed = self.np_random.uniform(*self.wind_speed_range)
        self.wind_direction = self.np_random.uniform(*self.wind_dir_range)

        self.yaw_angles = np.zeros(self.n_turbines)
        self.prev_yaw_angles = np.zeros(self.n_turbines)
        self.turbine_powers = self._compute_powers()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        self.prev_yaw_angles = self.yaw_angles.copy()

        # Absolute yaw control: action in [-1,1] -> yaw in [-max_yaw, max_yaw]
        self.yaw_angles = action * self.max_yaw

        # Evolve wind conditions
        self.wind_speed += self.np_random.normal(0, self.wind_variability * 0.1)
        self.wind_speed = np.clip(self.wind_speed, *self.wind_speed_range)
        self.wind_direction += self.np_random.normal(0, self.dir_variability * 0.1)
        self.wind_direction = np.clip(self.wind_direction, *self.wind_dir_range)

        # Compute power
        self.turbine_powers = self._compute_powers()
        farm_power = self.turbine_powers.sum()

        # ---- Reward ----
        # Power reward: normalized by rated farm power
        power_reward = farm_power / self.rated_farm_power

        # Yaw actuation cost: penalize absolute yaw magnitude (discourages unnecessary yaw)
        yaw_magnitude_cost = np.mean(np.abs(self.yaw_angles)) / self.max_yaw  # [0, 1]
        # Yaw change cost: penalize rapid changes
        yaw_change = np.abs(self.yaw_angles - self.prev_yaw_angles)
        yaw_change_cost = np.mean(yaw_change) / self.max_yaw  # [0, 2]

        reward = power_reward - self.c_yaw * (0.5 * yaw_magnitude_cost + 0.5 * yaw_change_cost)

        terminated = self.current_step >= self.episode_length
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        ws_norm = (self.wind_speed - self.wind_speed_range[0]) / \
                  (self.wind_speed_range[1] - self.wind_speed_range[0])
        wd_norm = (self.wind_direction - self.wind_dir_range[0]) / \
                  (self.wind_dir_range[1] - self.wind_dir_range[0])
        yaw_norm = self.yaw_angles / self.max_yaw
        power_norm = self.turbine_powers / self.rated_power

        return np.concatenate([
            [ws_norm, wd_norm],
            self.norm_x,
            self.norm_y,
            yaw_norm,
            power_norm,
        ]).astype(np.float32)

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
        """Power with all yaw=0 (greedy baseline)."""
        saved_yaw = self.yaw_angles.copy()
        self.yaw_angles = np.zeros(self.n_turbines)
        powers = self._compute_powers()
        self.yaw_angles = saved_yaw
        return powers.sum()
