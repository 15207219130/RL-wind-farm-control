"""
Train MAPPO (Multi-Agent PPO with parameter sharing) for wind farm yaw control.

Uses SharedPolicyMAWrapper to adapt PettingZoo ParallelEnv for SB3.
Each turbine is an independent agent with shared policy weights (CTDE).
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from src.envs.wind_farm_ma_env import WindFarmMAEnv
from src.envs.ma_sb3_wrapper import SharedPolicyMAWrapper


class PowerTrackingCallback(BaseCallback):
    def __init__(self, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.power_history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            infos = self.locals.get("infos", [])
            powers = [info.get("farm_power_mw", 0) for info in infos if "farm_power_mw" in info]
            if powers:
                avg_power = np.mean(powers)
                self.power_history.append((self.num_timesteps, avg_power))
                if self.verbose:
                    print(f"  Step {self.num_timesteps}: avg_power={avg_power:.3f}MW")
        return True


def make_env(seed=0, **kwargs):
    def _init():
        env = SharedPolicyMAWrapper(WindFarmMAEnv, **kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate_mappo(model, env_kwargs, n_episodes=20):
    """Evaluate trained MAPPO policy vs greedy baseline."""
    from floris import FlorisModel

    rl_powers = []
    greedy_powers = []
    yaw_stats = []

    for ep in range(n_episodes):
        env = SharedPolicyMAWrapper(WindFarmMAEnv, **env_kwargs)
        obs, info = env.reset(seed=ep + 1000)

        ep_rl_power = []
        ep_greedy_power = []
        ep_yaws = []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_rl_power.append(info.get("farm_power_mw", 0))
            ep_yaws.append(info.get("yaw_angles", np.zeros(9)))

            # Greedy baseline: get power with yaw=0
            inner_env = env.env
            inner_env.fm.set(
                layout_x=inner_env.layout_x,
                layout_y=inner_env.layout_y,
                wind_speeds=[inner_env.wind_speed],
                wind_directions=[inner_env.wind_direction],
                turbulence_intensities=[inner_env.turbulence_intensity],
                yaw_angles=np.zeros((1, inner_env.n_turbines)),
            )
            inner_env.fm.run()
            ep_greedy_power.append(inner_env.fm.get_turbine_powers().sum() / 1e6)

        rl_powers.append(np.mean(ep_rl_power))
        greedy_powers.append(np.mean(ep_greedy_power))
        yaw_stats.append(np.mean([np.abs(y).mean() for y in ep_yaws]))

    rl_avg = np.mean(rl_powers)
    greedy_avg = np.mean(greedy_powers)
    improvement = (rl_avg - greedy_avg) / greedy_avg * 100

    return {
        "rl_power_mw": rl_avg,
        "greedy_power_mw": greedy_avg,
        "improvement_pct": improvement,
        "avg_abs_yaw_deg": np.mean(yaw_stats),
        "n_episodes": n_episodes,
    }


def main():
    total_timesteps = 100_000
    n_envs = 4
    results_dir = Path("d:/work/code/RL for wind turbine control/results/mappo_baseline")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wind Farm RL Control - MAPPO (Parameter Sharing) Training")
    print("=" * 60)

    env_kwargs = dict(
        episode_length=200,
        wind_speed_range=(5.0, 15.0),
        wind_dir_range=(250.0, 290.0),
    )

    # Create vectorized training environments
    train_envs = DummyVecEnv([make_env(seed=i, **env_kwargs) for i in range(n_envs)])

    # PPO with larger network for multi-agent (9 agents concatenated)
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[512, 256, 256]),
        verbose=1,
        seed=42,
    )

    power_cb = PowerTrackingCallback(eval_freq=5000, verbose=1)

    print(f"\nTraining MAPPO for {total_timesteps} timesteps with {n_envs} parallel envs...")
    print(f"Network: MLP [512, 256, 256]")
    print(f"Observation dim: {train_envs.observation_space.shape}")
    print(f"Action dim: {train_envs.action_space.shape}")
    print()

    model.learn(total_timesteps=total_timesteps, callback=power_cb, progress_bar=True)

    # Save model
    model_path = results_dir / "mappo_wind_farm"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Evaluate
    print("\nEvaluating MAPPO policy vs greedy baseline...")
    results = evaluate_mappo(model, env_kwargs, n_episodes=20)

    print(f"\n{'='*60}")
    print(f"MAPPO EVALUATION RESULTS (20 episodes)")
    print(f"{'='*60}")
    print(f"  RL policy avg power:     {results['rl_power_mw']:.3f} MW")
    print(f"  Greedy baseline power:   {results['greedy_power_mw']:.3f} MW")
    print(f"  Improvement:             {results['improvement_pct']:+.2f}%")
    print(f"  Avg |yaw| angle:         {results['avg_abs_yaw_deg']:.1f}°")
    print(f"{'='*60}")

    # Save results
    results["timestamp"] = datetime.now().isoformat()
    results["total_timesteps"] = total_timesteps
    results["power_history"] = [(int(s), float(p)) for s, p in power_cb.power_history]

    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    train_envs.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
