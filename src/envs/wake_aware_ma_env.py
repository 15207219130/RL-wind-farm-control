"""
Wake-Aware Multi-Agent Wind Farm Environment.

Extends WindFarmMAEnv with explicit upstream wake information in observations.
Each agent's observation includes info about which turbines are upstream and
the estimated wake deficit they impose, making wake interactions explicit.

Observation layout (40 dims per agent):
  [ws, wd, ti, pos_x, pos_y, my_yaw, my_power,  # 7: local
   all_yaws(9), all_powers(9),                   # 18: global context
   upstream_0(5), upstream_1(5), upstream_2(5)]  # 15: upstream wake info

Each upstream slot: [dist_down_norm, dist_cross_norm, yaw_norm, power_norm, deficit]

Global state for centralized critic: 9 * 40 = 360 dims.
"""

import functools
import numpy as np
from gymnasium import spaces

from src.envs.wind_farm_ma_env import WindFarmMAEnv

# Upstream neighbor parameters
K_UPSTREAM = 3          # max upstream neighbors per agent
JENSEN_K = 0.075        # Jensen wake decay constant
CT = 0.8                # thrust coefficient (approximate)
YAW_PP = 1.88           # yaw power loss exponent
UPSTREAM_SLOT_DIM = 5   # [dist_down, dist_cross, yaw, power, deficit]


class WakeAwareMAEnv(WindFarmMAEnv):
    """
    Multi-agent wind farm environment with explicit wake-topology observations.

    Inherits all FLORIS physics from WindFarmMAEnv.
    Overrides observation_space() and _get_agent_obs() to add upstream info.
    """

    # Base obs dims from parent (without fatigue): 7 + 9*2 = 25
    _BASE_OBS_DIM = 7 + 9 * 2  # local(7) + all_yaws(9) + all_powers(9)
    _WAKE_OBS_DIM = _BASE_OBS_DIM + K_UPSTREAM * UPSTREAM_SLOT_DIM  # 25 + 15 = 40

    def __init__(self, **kwargs):
        # Disable fatigue for this env (keep things clean)
        kwargs.setdefault("enable_fatigue", False)
        super().__init__(**kwargs)

        # Normalization scale for distances
        spacing = self.spacing_D * self.rotor_diameter
        self._max_dist = (max(self.n_rows, self.n_cols) - 1) * spacing * np.sqrt(2) + 1e-6

        # Layout arrays as numpy for vectorized math
        self._lx = np.array(self.layout_x)
        self._ly = np.array(self.layout_y)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._WAKE_OBS_DIM,),
            dtype=np.float32,
        )

    def global_state_dim(self) -> int:
        """Dimension of concatenated global state (for centralized critic)."""
        return self.n_turbines * self._WAKE_OBS_DIM

    # ------------------------------------------------------------------
    # Upstream wake computation (Jensen analytical, wind-aligned frame)
    # ------------------------------------------------------------------

    def _compute_upstream_info(self, agent_idx: int) -> np.ndarray:
        """
        For turbine `agent_idx`, find up to K upstream turbines and return
        a (K * 5,) array with wake information for each.

        Steps:
        1. Rotate positions to wind-aligned frame (downwind / crosswind axes).
        2. Find turbines j where agent i is downwind of j.
        3. Filter by Jensen wake cone.
        4. Compute analytical wake deficit from j at i's position.
        5. Sort by deficit descending, take top K.
        6. Pad with zeros if < K candidates.

        Returns flat array of shape (K * UPSTREAM_SLOT_DIM,).
        """
        wd_deg = self.wind_direction
        # Wind direction in FLORIS convention: 270° = wind blowing in +x direction
        # Convert to math angle for coordinate rotation
        # If wind comes FROM 270° (west), it blows eastward (+x).
        # Wind-aligned frame: downwind axis = direction wind blows TO.
        wd_rad = np.radians(270.0 - wd_deg)  # angle of wind vector in standard math coords

        cos_wd = np.cos(wd_rad)
        sin_wd = np.sin(wd_rad)

        xi = self._lx[agent_idx]
        yi = self._ly[agent_idx]

        result = np.zeros((K_UPSTREAM, UPSTREAM_SLOT_DIM), dtype=np.float32)
        candidates = []

        for j in range(self.n_turbines):
            if j == agent_idx:
                continue

            xj, yj = self._lx[j], self._ly[j]
            dx = xi - xj  # vector from j to i
            dy = yi - yj

            # Project onto wind-aligned axes
            # downwind (how far i is downwind of j)
            dist_down = dx * cos_wd + dy * sin_wd
            # crosswind offset
            dist_cross = -dx * sin_wd + dy * cos_wd

            if dist_down <= 0:
                # j is not upstream of i
                continue

            # Jensen wake cone check: |crosswind| < rotor_radius + k * dist_down
            wake_radius = self.rotor_diameter / 2 + JENSEN_K * dist_down
            if abs(dist_cross) >= wake_radius:
                continue

            # Jensen wake deficit at distance dist_down
            yaw_j_rad = np.radians(self.yaw_angles[j])
            ct_eff = CT * np.cos(yaw_j_rad) ** 2
            # deficit = (1 - sqrt(1 - Ct_eff)) * (D / (D + 2*k*x))^2
            denom = self.rotor_diameter + 2 * JENSEN_K * dist_down
            deficit = (1.0 - np.sqrt(max(1.0 - ct_eff, 0.0))) * (self.rotor_diameter / denom) ** 2

            candidates.append({
                "j": j,
                "dist_down": dist_down,
                "dist_cross": dist_cross,
                "deficit": deficit,
            })

        # Sort by deficit descending (most impactful upstream turbines first)
        candidates.sort(key=lambda c: c["deficit"], reverse=True)

        rated_per_turbine = self.rated_farm_power / self.n_turbines

        for slot, cand in enumerate(candidates[:K_UPSTREAM]):
            j = cand["j"]
            result[slot, 0] = cand["dist_down"] / self._max_dist          # normalized downwind dist
            result[slot, 1] = cand["dist_cross"] / self._max_dist         # normalized crosswind offset
            result[slot, 2] = self.yaw_angles[j] / self.max_yaw           # upstream yaw (normalized)
            result[slot, 3] = self.turbine_powers[j] / rated_per_turbine  # upstream power (normalized)
            result[slot, 4] = float(cand["deficit"])                       # wake deficit [0, 1]

        return result.flatten()

    # ------------------------------------------------------------------
    # Override observation builder
    # ------------------------------------------------------------------

    def _get_agent_obs(self, agent) -> np.ndarray:
        idx = self.agent_name_to_idx[agent]

        # --- Normalized wind conditions ---
        ws_norm = (self.wind_speed - self.wind_speed_range[0]) / (
            self.wind_speed_range[1] - self.wind_speed_range[0]
        )
        wd_norm = (self.wind_direction - self.wind_dir_range[0]) / (
            self.wind_dir_range[1] - self.wind_dir_range[0]
        )
        ti_norm = (self.turbulence_intensity - self.ti_range[0]) / (
            self.ti_range[1] - self.ti_range[0]
        )

        # --- Local state ---
        my_yaw_norm = self.yaw_angles[idx] / self.max_yaw
        rated_per_turbine = self.rated_farm_power / self.n_turbines
        my_power_norm = self.turbine_powers[idx] / rated_per_turbine

        # --- Global context ---
        all_yaws_norm = self.yaw_angles / self.max_yaw
        all_powers_norm = self.turbine_powers / rated_per_turbine

        # --- Upstream wake info (15 dims) ---
        upstream_info = self._compute_upstream_info(idx)

        obs = np.concatenate([
            [ws_norm, wd_norm, ti_norm],          # 3
            [self.norm_x[idx], self.norm_y[idx]], # 2
            [my_yaw_norm, my_power_norm],          # 2
            all_yaws_norm,                         # 9
            all_powers_norm,                       # 9
            upstream_info,                         # 15
        ])

        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience: build global state tensor from current observations
    # ------------------------------------------------------------------

    def get_global_state(self) -> np.ndarray:
        """Return concatenation of all agents' observations (for centralized critic)."""
        obs_list = [self._get_agent_obs(agent) for agent in self.possible_agents]
        return np.concatenate(obs_list).astype(np.float32)
