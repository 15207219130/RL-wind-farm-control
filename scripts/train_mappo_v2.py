"""
Train Wake-Aware MAPPO for Wind Farm Yaw Control.

Uses:
  - WakeAwareMAEnv: FLORIS backend + explicit upstream wake observations (40 dims/agent)
  - MAPPOAgent: custom PyTorch MAPPO with decentralized actor + centralized critic (true CTDE)

Training loop:
  For each iteration:
    1. Collect one rollout (episode_length steps) from WakeAwareMAEnv
    2. Compute GAE advantages using centralized value function
    3. Run n_epochs of mini-batch PPO updates on actor and critic
    4. Log metrics; evaluate vs greedy baseline every eval_interval iterations

Outputs → results/mappo_v2/
  mappo_v2_actor.pt, mappo_v2_critic.pt
  training_log.json
  evaluation_results.json
  training_curves.png
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs.wake_aware_ma_env import WakeAwareMAEnv
from src.agents.mappo import MAPPOAgent, RolloutBuffer


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #

def evaluate(agent: MAPPOAgent, env_kwargs: dict, n_episodes: int = 20) -> dict:
    """
    Run deterministic episodes and compare against greedy baseline (yaw=0).
    """
    rl_powers, greedy_powers, yaw_stats = [], [], []

    for ep in range(n_episodes):
        env = WakeAwareMAEnv(**env_kwargs)
        obs_dict, _ = env.reset(seed=ep + 2000)

        ep_rl_power, ep_greedy, ep_yaws = [], [], []
        done = False

        while not done:
            # Collect per-agent observations
            obs_arr = np.stack([obs_dict[a] for a in env.possible_agents])  # (N, obs_dim)
            actions_arr, _ = agent.select_actions(obs_arr, deterministic=True)

            # Build action dict
            action_dict = {
                agent_id: actions_arr[i]
                for i, agent_id in enumerate(env.possible_agents)
            }

            obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)
            done = any(terminations.values()) or any(truncations.values())

            info = infos[env.possible_agents[0]]
            ep_rl_power.append(info.get("farm_power_mw", 0.0))
            ep_yaws.append(info.get("yaw_angles", np.zeros(env.n_turbines)).copy())

            # Greedy baseline: zero yaw via FLORIS
            env.fm.set(
                layout_x=env.layout_x,
                layout_y=env.layout_y,
                wind_speeds=[env.wind_speed],
                wind_directions=[env.wind_direction],
                turbulence_intensities=[env.turbulence_intensity],
                yaw_angles=np.zeros((1, env.n_turbines)),
            )
            env.fm.run()
            ep_greedy.append(env.fm.get_turbine_powers().sum() / 1e6)

        rl_powers.append(float(np.mean(ep_rl_power)))
        greedy_powers.append(float(np.mean(ep_greedy)))
        yaw_stats.append(float(np.mean([np.abs(y).mean() for y in ep_yaws])))

    rl_avg = float(np.mean(rl_powers))
    greedy_avg = float(np.mean(greedy_powers))
    improvement = (rl_avg - greedy_avg) / greedy_avg * 100 if greedy_avg > 0 else 0.0

    return {
        "rl_power_mw": rl_avg,
        "greedy_power_mw": greedy_avg,
        "improvement_pct": improvement,
        "avg_abs_yaw_deg": float(np.mean(yaw_stats)),
        "n_episodes": n_episodes,
        "per_episode_rl": rl_powers,
        "per_episode_greedy": greedy_powers,
    }


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

def plot_training(log: list[dict], eval_checkpoints: list[dict], save_path: Path):
    iters = [e["iteration"] for e in log]
    powers = [e["mean_farm_power_mw"] for e in log]
    actor_losses = [e["actor_loss"] for e in log]
    critic_losses = [e["critic_loss"] for e in log]
    entropies = [e["entropy"] for e in log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Wake-Aware MAPPO — Training Progress", fontsize=14)

    axes[0, 0].plot(iters, powers, color="steelblue", linewidth=1)
    axes[0, 0].set_title("Farm Power (MW)")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, actor_losses, color="coral", linewidth=1)
    axes[0, 1].set_title("Actor Loss")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, critic_losses, color="seagreen", linewidth=1)
    axes[1, 0].set_title("Critic Loss")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, entropies, color="mediumpurple", linewidth=1)
    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].grid(True, alpha=0.3)

    if eval_checkpoints:
        eval_iters = [c["iteration"] for c in eval_checkpoints]
        eval_improvements = [c["improvement_pct"] for c in eval_checkpoints]
        ax2 = axes[0, 0].twinx()
        ax2.plot(eval_iters, eval_improvements, "r--o", markersize=4, label="Improvement %")
        ax2.set_ylabel("Improvement over greedy (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ #
# Main training loop
# ------------------------------------------------------------------ #

def main():
    # ---- Hyperparameters ----
    N_ITERATIONS = 500       # total rollout iterations
    ROLLOUT_LEN = 200        # steps per rollout (= episode length)
    EVAL_INTERVAL = 50       # evaluate every N iterations
    LOG_INTERVAL = 10        # print progress every N iterations

    RESULTS_DIR = Path("d:/work/code/RL for wind turbine control/results/mappo_v2")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ENV_KWARGS = dict(
        episode_length=ROLLOUT_LEN,
        wind_speed_range=(5.0, 15.0),
        wind_dir_range=(250.0, 290.0),
        ti_range=(0.04, 0.10),
        enable_fatigue=False,
    )

    AGENT_KWARGS = dict(
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        mini_batch_size=256,
        device="cpu",
    )

    # ---- Environment + Agent setup ----
    env = WakeAwareMAEnv(**ENV_KWARGS)
    obs_dim = env._WAKE_OBS_DIM          # 40
    act_dim = 1
    n_agents = env.n_turbines            # 9
    global_state_dim = env.global_state_dim()  # 360

    agent = MAPPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_agents=n_agents,
        global_state_dim=global_state_dim,
        **AGENT_KWARGS,
    )

    buffer = RolloutBuffer()

    print("=" * 65)
    print("  Wake-Aware MAPPO — Wind Farm Yaw Control")
    print("=" * 65)
    print(f"  Turbines       : {n_agents} (3×3 grid)")
    print(f"  Obs dim/agent  : {obs_dim}  (25 base + 15 wake info)")
    print(f"  Global state   : {global_state_dim}")
    print(f"  Act dim/agent  : {act_dim}")
    print(f"  Iterations     : {N_ITERATIONS}  ×  {ROLLOUT_LEN} steps")
    print(f"  Total steps    : {N_ITERATIONS * ROLLOUT_LEN:,}")
    print(f"  Results dir    : {RESULTS_DIR}")
    print("=" * 65)
    print()

    training_log = []
    eval_checkpoints = []

    # ---- Training ----
    for iteration in range(N_ITERATIONS):
        buffer.clear()
        obs_dict, _ = env.reset(seed=iteration)

        episode_rewards = []
        episode_powers = []

        for t in range(ROLLOUT_LEN):
            # Collect observations for all agents
            obs_arr = np.stack([obs_dict[a] for a in env.possible_agents])  # (N, obs_dim)
            global_state = env.get_global_state()  # (global_state_dim,)

            # Get centralized value
            value = agent.get_value(global_state)

            # Sample actions
            actions_arr, log_probs_arr = agent.select_actions(obs_arr)

            # Build action dict for environment (each agent gets shape-(act_dim,) array)
            action_dict = {
                agent_id: actions_arr[i]
                for i, agent_id in enumerate(env.possible_agents)
            }

            # Environment step
            next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(action_dict)

            rewards_arr = np.array([rewards_dict[a] for a in env.possible_agents])
            done = any(terminations.values()) or any(truncations.values())

            buffer.add(
                obs=obs_arr,
                global_state=global_state,
                actions=actions_arr,
                log_probs=log_probs_arr,
                rewards=rewards_arr,
                value=value,
                done=done,
            )

            info = infos[env.possible_agents[0]]
            episode_rewards.append(float(rewards_arr.mean()))
            episode_powers.append(info.get("farm_power_mw", 0.0))

            obs_dict = next_obs_dict

            if done:
                break

        # Bootstrap value for last state
        last_gs = env.get_global_state()
        last_value = agent.get_value(last_gs) if not done else 0.0

        # PPO update
        losses = agent.update(buffer, last_value=last_value)

        # Logging
        mean_power = float(np.mean(episode_powers))
        mean_reward = float(np.mean(episode_rewards))

        log_entry = {
            "iteration": iteration,
            "mean_farm_power_mw": mean_power,
            "mean_reward": mean_reward,
            **losses,
        }
        training_log.append(log_entry)

        if iteration % LOG_INTERVAL == 0:
            print(
                f"  Iter {iteration:4d}/{N_ITERATIONS}  "
                f"power={mean_power:.3f}MW  "
                f"actor_loss={losses['actor_loss']:.4f}  "
                f"critic_loss={losses['critic_loss']:.4f}  "
                f"entropy={losses['entropy']:.4f}"
            )

        # Periodic evaluation
        if (iteration + 1) % EVAL_INTERVAL == 0:
            print(f"\n  [Eval @ iter {iteration + 1}]", end=" ")
            eval_res = evaluate(agent, ENV_KWARGS, n_episodes=10)
            print(
                f"RL={eval_res['rl_power_mw']:.3f}MW  "
                f"greedy={eval_res['greedy_power_mw']:.3f}MW  "
                f"improvement={eval_res['improvement_pct']:+.2f}%\n"
            )
            eval_checkpoints.append({
                "iteration": iteration + 1,
                **eval_res,
            })

            # Save checkpoint
            agent.save(str(RESULTS_DIR))

    # ---- Final evaluation ----
    print("\n" + "=" * 65)
    print("  Final Evaluation (20 episodes)")
    print("=" * 65)
    final_results = evaluate(agent, ENV_KWARGS, n_episodes=20)
    print(f"  RL avg power   : {final_results['rl_power_mw']:.3f} MW")
    print(f"  Greedy power   : {final_results['greedy_power_mw']:.3f} MW")
    print(f"  Improvement    : {final_results['improvement_pct']:+.2f}%")
    print(f"  Avg |yaw|      : {final_results['avg_abs_yaw_deg']:.1f}°")

    # ---- Save results ----
    agent.save(str(RESULTS_DIR))

    final_results["timestamp"] = datetime.now().isoformat()
    final_results["n_iterations"] = N_ITERATIONS
    final_results["rollout_len"] = ROLLOUT_LEN
    final_results["total_steps"] = N_ITERATIONS * ROLLOUT_LEN

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    with open(RESULTS_DIR / "training_log.json", "w") as f:
        json.dump({"log": training_log, "eval_checkpoints": eval_checkpoints}, f, indent=2)

    plot_training(training_log, eval_checkpoints, RESULTS_DIR / "training_curves.png")

    print(f"\n  Saved to: {RESULTS_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
