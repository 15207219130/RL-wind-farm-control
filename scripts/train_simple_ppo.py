"""
Train PPO on SimpleWindFarmEnv: maximize power with yaw control cost.

Usage:
    python scripts/train_simple_ppo.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.envs.simple_wind_farm_env import SimpleWindFarmEnv


# ============================================================
# Callback
# ============================================================
class TrainingLogger(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, n_eval=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.history = {"steps": [], "power_mw": [], "greedy_mw": [],
                        "improvement_pct": [], "avg_yaw": [], "reward": []}

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            powers, greedys, yaws, rews = [], [], [], []
            for ep in range(self.n_eval):
                obs, info = self.eval_env.reset(seed=ep + 5000)
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = self.eval_env.step(action)
                    done = term or trunc
                    powers.append(info["farm_power_mw"])
                    greedys.append(self.eval_env.get_greedy_power() / 1e6)
                    yaws.append(np.abs(info["yaw_angles"]).mean())
                    rews.append(r)

            avg_p = np.mean(powers)
            avg_g = np.mean(greedys)
            imp = (avg_p - avg_g) / avg_g * 100 if avg_g > 0 else 0

            self.history["steps"].append(self.num_timesteps)
            self.history["power_mw"].append(float(avg_p))
            self.history["greedy_mw"].append(float(avg_g))
            self.history["improvement_pct"].append(float(imp))
            self.history["avg_yaw"].append(float(np.mean(yaws)))
            self.history["reward"].append(float(np.mean(rews)))

            if self.verbose:
                print(f"  [{self.num_timesteps:>7d}]  "
                      f"power={avg_p:.2f} MW  greedy={avg_g:.2f} MW  "
                      f"imp={imp:+.2f}%  |yaw|={np.mean(yaws):.1f}°")
        return True


# ============================================================
# Final evaluation
# ============================================================
def evaluate(model, env, n_episodes=30):
    rl_powers, greedy_powers, yaw_means = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 9000)
        ep_rl, ep_greedy, ep_yaw = [], [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rl.append(info["farm_power_mw"])
            ep_greedy.append(env.get_greedy_power() / 1e6)
            ep_yaw.append(info["yaw_angles"].copy())

        rl_powers.append(np.mean(ep_rl))
        greedy_powers.append(np.mean(ep_greedy))
        yaw_means.append(np.mean([np.abs(y).mean() for y in ep_yaw]))

    rl_avg = np.mean(rl_powers)
    gr_avg = np.mean(greedy_powers)
    return {
        "rl_power_mw": float(rl_avg),
        "greedy_power_mw": float(gr_avg),
        "improvement_pct": float((rl_avg - gr_avg) / gr_avg * 100),
        "avg_abs_yaw_deg": float(np.mean(yaw_means)),
        "rl_power_std": float(np.std(rl_powers)),
        "greedy_power_std": float(np.std(greedy_powers)),
    }


# ============================================================
# Main
# ============================================================
def main():
    total_timesteps = 1_000_000
    n_envs = 8
    c_yaw = 0.02

    results_dir = Path("d:/work/code/RL for wind turbine control/results/simple_ppo")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Simple Wind Farm RL - PPO with Yaw Cost")
    print(f"  c_yaw = {c_yaw}  |  steps = {total_timesteps}")
    print("=" * 60)

    env_kw = dict(
        n_rows=3,
        n_cols=3,
        episode_length=200,
        wind_speed_range=(7.0, 12.0),
        wind_dir_range=(265.0, 275.0),
        c_yaw=c_yaw,
    )

    def make_env(seed):
        def _init():
            env = SimpleWindFarmEnv(**env_kw)
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    train_envs = DummyVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = SimpleWindFarmEnv(**env_kw)

    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        seed=42,
    )

    logger = TrainingLogger(eval_env, eval_freq=10000, n_eval=5, verbose=1)

    print(f"\nTraining with {n_envs} parallel envs ...\n")
    model.learn(total_timesteps=total_timesteps, callback=logger, progress_bar=True)

    model_path = results_dir / "ppo_simple_wind_farm"
    model.save(str(model_path))
    print(f"\nModel saved: {model_path}")

    # Evaluate
    print("\nFinal evaluation (30 episodes) ...")
    results = evaluate(model, eval_env)

    print(f"\n{'=' * 60}")
    print(f"  RL power:      {results['rl_power_mw']:.3f} +/- {results['rl_power_std']:.3f} MW")
    print(f"  Greedy power:  {results['greedy_power_mw']:.3f} +/- {results['greedy_power_std']:.3f} MW")
    print(f"  Improvement:   {results['improvement_pct']:+.2f}%")
    print(f"  Avg |yaw|:     {results['avg_abs_yaw_deg']:.1f}°")
    print(f"{'=' * 60}")

    # Save
    results["timestamp"] = datetime.now().isoformat()
    results["total_timesteps"] = total_timesteps
    results["c_yaw"] = c_yaw
    results["training_history"] = logger.history
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    train_envs.close()

    # Plot
    plot_results(logger.history, results, results_dir)
    print(f"Plots saved to {results_dir}")


def plot_results(history, results, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = history["steps"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Power comparison over training
    ax = axes[0, 0]
    ax.plot(steps, history["power_mw"], "b-", lw=1.5, label="RL policy")
    ax.plot(steps, history["greedy_mw"], "r--", lw=1.5, label="Greedy (yaw=0)")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Farm Power (MW)")
    ax.set_title("Power During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Improvement %
    ax = axes[0, 1]
    ax.plot(steps, history["improvement_pct"], "g-", lw=1.5)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Improvement over Greedy (%)")
    ax.set_title("Wake Steering Gain")
    ax.grid(True, alpha=0.3)

    # Yaw usage
    ax = axes[1, 0]
    ax.plot(steps, history["avg_yaw"], "m-", lw=1.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Avg |Yaw| (deg)")
    ax.set_title("Yaw Angle Usage")
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[1, 1]
    ax.plot(steps, history["reward"], color="darkorange", lw=1.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title(f"Reward (c_yaw={results['c_yaw']})")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Wind Farm RL Control - PPO Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    x = ["Greedy\n(yaw=0)", "RL\n(PPO)"]
    vals = [results["greedy_power_mw"], results["rl_power_mw"]]
    errs = [results["greedy_power_std"], results["rl_power_std"]]
    colors = ["#d9534f", "#5cb85c"]
    bars = ax.bar(x, vals, yerr=errs, capsize=5, color=colors, width=0.5)
    ax.set_ylabel("Avg Farm Power (MW)")
    imp = results["improvement_pct"]
    ax.set_title(f"Power Comparison ({imp:+.1f}%)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "power_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
