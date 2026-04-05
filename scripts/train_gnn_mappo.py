"""
Train GNN-MAPPO for Wind Farm Yaw Control — Parallel Rollout Edition.

Each training iteration spawns N_PARALLEL_ENVS worker processes.
Every worker runs a full episode independently, using only the actor
(no critic in workers → saves memory, avoids pickling the critic).
The main process aggregates all trajectories, computes values + GAE
using the centralized critic, then runs the PPO update.

Speedup: roughly linear with N_PARALLEL_ENVS up to available CPU cores,
because the bottleneck is FLORIS simulation (CPU-bound per-step).

Outputs → results/gnn_mappo/
  gnn_mappo_actor.pt, gnn_mappo_critic.pt
  training_log.json, evaluation_results.json, training_curves.png
"""

import os
# Prevent OpenBLAS/MKL from spawning extra threads inside each worker.
# Must be set BEFORE numpy is imported anywhere in the process.
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs.wake_aware_ma_env import WakeAwareMAEnv
from src.models.gnn import WakeFarmGraph
from src.agents.gnn_mappo import GNNMAPPOAgent, GNNActorNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Worker (must be at module level for multiprocessing pickling)
# ─────────────────────────────────────────────────────────────────────────────

def rollout_worker(args):
    """
    Run one full episode and return the trajectory as numpy arrays.

    Only the actor is needed here — the critic stays in the main process.
    This function runs inside a subprocess (spawned by ProcessPoolExecutor).

    Returns dict with keys:
      node_feat (T,N,d), adj (T,N,N), edge_feat (T,N,N,d),
      actions (T,N,act), log_probs (T,N), rewards (T,N), dones (T,),
      last_X (N,d), last_A (N,N), last_E (N,N,d), last_done bool,
      mean_power float  (for logging)
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS",     "1")
    os.environ.setdefault("MKL_NUM_THREADS",     "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    import torch
    seed, env_kwargs, graph_kwargs, actor_bytes, actor_cfg = args

    # ── Rebuild environment ──────────────────────────────────────────────────
    env = WakeAwareMAEnv(**env_kwargs)
    gb  = WakeFarmGraph(**graph_kwargs)

    # ── Rebuild actor from serialised weights ────────────────────────────────
    actor = GNNActorNetwork(
        act_dim   = actor_cfg["act_dim"],
        embed_dim = actor_cfg["embed_dim"],
        n_layers  = actor_cfg["n_layers"],
    )
    actor.load_state_dict(torch.load(io.BytesIO(actor_bytes), map_location="cpu",
                                     weights_only=True))
    actor.eval()

    # ── Run episode ──────────────────────────────────────────────────────────
    env.reset(seed=seed)

    traj = {k: [] for k in ("node_feat", "adj", "edge_feat",
                             "actions", "log_probs", "rewards", "dones")}
    powers = []

    for _ in range(env_kwargs["episode_length"]):
        X, A, E = gb.build(env.wind_speed, env.wind_direction,
                           env.yaw_angles, env.turbine_powers)

        with torch.no_grad():
            Xt = torch.tensor(X).unsqueeze(0)   # (1, N, d)
            At = torch.tensor(A).unsqueeze(0)
            Et = torch.tensor(E).unsqueeze(0)
            acts_t, lp_t = actor.get_action(Xt, At, Et)
            acts_np = acts_t.squeeze(0).numpy()   # (N, act_dim)
            lp_np   = lp_t.squeeze(0).numpy()     # (N,)

        action_dict = {a: acts_np[i] for i, a in enumerate(env.possible_agents)}
        _, rew_dict, terms, truncs, infos = env.step(action_dict)

        rews = np.array([rew_dict[a] for a in env.possible_agents], dtype=np.float32)
        done = any(terms.values()) or any(truncs.values())

        traj["node_feat"].append(X)
        traj["adj"].append(A)
        traj["edge_feat"].append(E)
        traj["actions"].append(acts_np)
        traj["log_probs"].append(lp_np)
        traj["rewards"].append(rews)
        traj["dones"].append(done)
        powers.append(infos[env.possible_agents[0]].get("farm_power_mw", 0.0))

        if done:
            break

    # Last state for bootstrap value
    last_X, last_A, last_E = gb.build(env.wind_speed, env.wind_direction,
                                       env.yaw_angles, env.turbine_powers)

    return {
        "node_feat":  np.stack(traj["node_feat"]),   # (T, N, d_node)
        "adj":        np.stack(traj["adj"]),          # (T, N, N)
        "edge_feat":  np.stack(traj["edge_feat"]),    # (T, N, N, d_edge)
        "actions":    np.stack(traj["actions"]),      # (T, N, act_dim)
        "log_probs":  np.stack(traj["log_probs"]),    # (T, N)
        "rewards":    np.stack(traj["rewards"]),      # (T, N)
        "dones":      np.array(traj["dones"], dtype=np.float32),  # (T,)
        "last_X":     last_X,
        "last_A":     last_A,
        "last_E":     last_E,
        "last_done":  bool(traj["dones"][-1]),
        "mean_power": float(np.mean(powers)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent, graph_builder, env_kwargs, n_episodes=20):
    """Deterministic evaluation vs greedy (yaw=0) baseline."""
    rl_powers, greedy_powers, yaw_stats = [], [], []

    for ep in range(n_episodes):
        env = WakeAwareMAEnv(**env_kwargs)
        env.reset(seed=ep + 2000)
        ep_rl, ep_greedy, ep_yaws = [], [], []
        done = False

        while not done:
            X, A, E = graph_builder.build(env.wind_speed, env.wind_direction,
                                          env.yaw_angles, env.turbine_powers)
            acts, _ = agent.select_actions(X, A, E, deterministic=True)
            action_dict = {a: acts[i] for i, a in enumerate(env.possible_agents)}
            _, _, terms, truncs, infos = env.step(action_dict)
            done = any(terms.values()) or any(truncs.values())

            info = infos[env.possible_agents[0]]
            ep_rl.append(info.get("farm_power_mw", 0.0))
            ep_yaws.append(info.get("yaw_angles", np.zeros(env.n_turbines)).copy())

            env.fm.set(layout_x=env.layout_x, layout_y=env.layout_y,
                       wind_speeds=[env.wind_speed], wind_directions=[env.wind_direction],
                       turbulence_intensities=[env.turbulence_intensity],
                       yaw_angles=np.zeros((1, env.n_turbines)))
            env.fm.run()
            ep_greedy.append(env.fm.get_turbine_powers().sum() / 1e6)

        rl_powers.append(float(np.mean(ep_rl)))
        greedy_powers.append(float(np.mean(ep_greedy)))
        yaw_stats.append(float(np.mean([np.abs(y).mean() for y in ep_yaws])))

    rl_avg      = float(np.mean(rl_powers))
    greedy_avg  = float(np.mean(greedy_powers))
    improvement = (rl_avg - greedy_avg) / max(greedy_avg, 1e-6) * 100

    return {"rl_power_mw": rl_avg, "greedy_power_mw": greedy_avg,
            "improvement_pct": improvement,
            "avg_abs_yaw_deg": float(np.mean(yaw_stats)),
            "n_episodes": n_episodes,
            "per_episode_rl": rl_powers, "per_episode_greedy": greedy_powers}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(log, eval_ckpts, save_path):
    iters     = [e["iteration"]          for e in log]
    powers    = [e["mean_farm_power_mw"] for e in log]
    a_losses  = [e["actor_loss"]         for e in log]
    c_losses  = [e["critic_loss"]        for e in log]
    entropies = [e["entropy"]            for e in log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GNN-MAPPO — Parallel Rollout Training", fontsize=14)

    axes[0,0].plot(iters, powers,    color="steelblue",    lw=1)
    axes[0,0].set_title("Farm Power (MW)"); axes[0,0].grid(alpha=.3)
    axes[0,1].plot(iters, a_losses,  color="coral",        lw=1)
    axes[0,1].set_title("Actor Loss"); axes[0,1].grid(alpha=.3)
    axes[1,0].plot(iters, c_losses,  color="seagreen",     lw=1)
    axes[1,0].set_title("Critic Loss"); axes[1,0].grid(alpha=.3)
    axes[1,1].plot(iters, entropies, color="mediumpurple", lw=1)
    axes[1,1].set_title("Policy Entropy"); axes[1,1].grid(alpha=.3)

    if eval_ckpts:
        ei   = [c["iteration"]       for c in eval_ckpts]
        eimp = [c["improvement_pct"] for c in eval_ckpts]
        ax2  = axes[0,0].twinx()
        ax2.plot(ei, eimp, "r--o", ms=4, label="Improvement %")
        ax2.set_ylabel("Improvement over greedy (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    for ax in axes.flat:
        ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Hyperparameters ──────────────────────────────────────────────────────
    N_ITERATIONS    = 500
    ROLLOUT_LEN     = 200
    N_PARALLEL_ENVS = 8      # ← number of parallel worker processes
    EVAL_INTERVAL   = 50
    LOG_INTERVAL    = 10

    RESULTS_DIR = Path("results/gnn_mappo")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ENV_KWARGS = dict(
        episode_length   = ROLLOUT_LEN,
        wind_speed_range = (5.0, 15.0),
        wind_dir_range   = (250.0, 290.0),
        ti_range         = (0.04, 0.10),
        enable_fatigue   = False,
        absolute_yaw     = True,
    )

    AGENT_KWARGS = dict(
        n_agents        = 9,
        act_dim         = 1,
        embed_dim       = 64,
        gnn_layers      = 3,
        lr_actor        = 3e-4,
        lr_critic       = 3e-4,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_eps        = 0.2,
        value_coef      = 0.5,
        entropy_coef    = 0.01,
        max_grad_norm   = 0.5,
        n_epochs        = 10,
        mini_batch_size = 256,
        device          = "cpu",
    )

    # actor config for workers (mirrors GNNActorNetwork constructor args)
    ACTOR_CFG = dict(
        act_dim   = AGENT_KWARGS["act_dim"],
        embed_dim = AGENT_KWARGS["embed_dim"],
        n_layers  = AGENT_KWARGS["gnn_layers"],
    )

    # ── One-time env to extract layout info ──────────────────────────────────
    _env = WakeAwareMAEnv(**ENV_KWARGS)
    _env.reset(seed=0)

    GRAPH_KWARGS = dict(
        layout_x                = list(_env.layout_x),
        layout_y                = list(_env.layout_y),
        rotor_diameter          = _env.rotor_diameter,
        max_yaw                 = _env.max_yaw,
        wind_speed_range        = _env.wind_speed_range,
        wind_dir_range          = _env.wind_dir_range,
        rated_power_per_turbine = float(_env.rated_farm_power / _env.n_turbines),
    )

    # ── Setup agent + graph builder (main process) ───────────────────────────
    agent         = GNNMAPPOAgent(**AGENT_KWARGS)
    graph_builder = WakeFarmGraph(**GRAPH_KWARGS)

    actor_params  = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())

    print("=" * 65)
    print("  GNN-MAPPO — Parallel Rollout Training")
    print("=" * 65)
    print(f"  Parallel envs  : {N_PARALLEL_ENVS}")
    print(f"  Steps/iter     : {N_PARALLEL_ENVS} × {ROLLOUT_LEN} = "
          f"{N_PARALLEL_ENVS * ROLLOUT_LEN}")
    print(f"  Total steps    : {N_ITERATIONS * N_PARALLEL_ENVS * ROLLOUT_LEN:,}")
    print(f"  Actor params   : {actor_params:,}")
    print(f"  Critic params  : {critic_params:,}")
    print(f"  Results dir    : {RESULTS_DIR}")
    print("=" * 65)
    print()

    training_log = []
    eval_ckpts   = []

    # ── Training loop ────────────────────────────────────────────────────────
    for iteration in range(N_ITERATIONS):

            actor_bytes = agent.serialize_actor()

            # Collect N episodes sequentially (FLORIS is not thread-safe)
            trajs = []
            for w in range(N_PARALLEL_ENVS):
                args = (iteration * N_PARALLEL_ENVS + w, ENV_KWARGS,
                        GRAPH_KWARGS, actor_bytes, ACTOR_CFG)
                trajs.append(rollout_worker(args))
                print(f"  Iter {iteration:4d}  rollout {w+1}/{N_PARALLEL_ENVS}"
                      f"  power={trajs[-1]['mean_power']:.2f}MW", flush=True)

            # PPO update using all collected trajectories
            losses = agent.update_from_trajs(trajs)

            mean_power = float(np.mean([t["mean_power"] for t in trajs]))

            log_entry = {"iteration": iteration,
                         "mean_farm_power_mw": mean_power, **losses}
            training_log.append(log_entry)

            if iteration % LOG_INTERVAL == 0:
                print(
                    f"  Iter {iteration:4d}/{N_ITERATIONS}  "
                    f"power={mean_power:.3f}MW  "
                    f"actor={losses['actor_loss']:.4f}  "
                    f"critic={losses['critic_loss']:.4f}  "
                    f"entropy={losses['entropy']:.4f}"
                )

            if (iteration + 1) % EVAL_INTERVAL == 0:
                print(f"\n  [Eval @ iter {iteration + 1}]", end=" ", flush=True)
                res = evaluate(agent, graph_builder, ENV_KWARGS, n_episodes=10)
                print(
                    f"RL={res['rl_power_mw']:.3f}MW  "
                    f"greedy={res['greedy_power_mw']:.3f}MW  "
                    f"improvement={res['improvement_pct']:+.2f}%\n"
                )
                eval_ckpts.append({"iteration": iteration + 1, **res})
                agent.save(str(RESULTS_DIR))

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Final Evaluation (20 episodes)")
    print("=" * 65)
    final = evaluate(agent, graph_builder, ENV_KWARGS, n_episodes=20)
    print(f"  RL avg power   : {final['rl_power_mw']:.3f} MW")
    print(f"  Greedy power   : {final['greedy_power_mw']:.3f} MW")
    print(f"  Improvement    : {final['improvement_pct']:+.2f}%")
    print(f"  Avg |yaw|      : {final['avg_abs_yaw_deg']:.1f}°")

    # ── Save ─────────────────────────────────────────────────────────────────
    agent.save(str(RESULTS_DIR))
    final.update({"timestamp": datetime.now().isoformat(),
                  "n_iterations": N_ITERATIONS, "rollout_len": ROLLOUT_LEN,
                  "n_parallel_envs": N_PARALLEL_ENVS,
                  "total_steps": N_ITERATIONS * N_PARALLEL_ENVS * ROLLOUT_LEN})
    (RESULTS_DIR / "evaluation_results.json").write_text(json.dumps(final, indent=2))
    (RESULTS_DIR / "training_log.json").write_text(
        json.dumps({"log": training_log, "eval_checkpoints": eval_ckpts}, indent=2))
    plot_training(training_log, eval_ckpts, RESULTS_DIR / "training_curves.png")
    print(f"\n  Saved to: {RESULTS_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
