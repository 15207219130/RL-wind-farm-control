"""
FLORIS gradient-based yaw optimizer baseline.

This uses FLORIS's built-in optimization to find optimal yaw angles
for each wind condition. Serves as the classical control baseline
against which RL methods are compared.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from floris import FlorisModel


def optimize_yaw_scipy(fm, wind_speed, wind_direction, ti, n_turbines, max_yaw=30.0):
    """Optimize yaw angles using scipy for a given wind condition."""

    def neg_farm_power(yaw_flat):
        fm.set(
            wind_speeds=[wind_speed],
            wind_directions=[wind_direction],
            turbulence_intensities=[ti],
            yaw_angles=yaw_flat.reshape(1, -1),
        )
        fm.run()
        return -fm.get_turbine_powers().sum()

    x0 = np.zeros(n_turbines)
    bounds = [(-max_yaw, max_yaw)] * n_turbines

    result = minimize(
        neg_farm_power, x0, method='SLSQP', bounds=bounds,
        options={'ftol': 1e-6, 'maxiter': 100}
    )

    return result.x, -result.fun


def main():
    results_dir = Path("d:/work/code/RL for wind turbine control/results/floris_optimizer")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wind Farm Control - FLORIS Gradient Optimizer Baseline")
    print("=" * 60)

    # Setup farm
    D = 126.0
    spacing = 5 * D
    layout_x, layout_y = [], []
    for i in range(3):
        for j in range(3):
            layout_x.append(i * spacing)
            layout_y.append(j * spacing)

    fm = FlorisModel(FlorisModel.get_defaults())
    fm.set(layout_x=layout_x, layout_y=layout_y)
    n_turbines = 9

    # Test across wind conditions
    wind_speeds = [6.0, 8.0, 10.0, 12.0, 14.0]
    wind_directions = [255.0, 265.0, 270.0, 275.0, 285.0]
    ti = 0.06

    results = []
    improvements = []

    print(f"\nOptimizing yaw angles for {len(wind_speeds)}x{len(wind_directions)} wind conditions...\n")

    for ws in wind_speeds:
        for wd in wind_directions:
            # Greedy baseline (yaw=0)
            fm.set(
                wind_speeds=[ws], wind_directions=[wd],
                turbulence_intensities=[ti],
                yaw_angles=np.zeros((1, n_turbines)),
            )
            fm.run()
            greedy_power = fm.get_turbine_powers().sum()

            # Optimized yaw
            opt_yaws, opt_power = optimize_yaw_scipy(fm, ws, wd, ti, n_turbines)
            improvement = (opt_power - greedy_power) / greedy_power * 100
            improvements.append(improvement)

            results.append({
                "wind_speed": ws,
                "wind_direction": wd,
                "greedy_power_mw": greedy_power / 1e6,
                "optimized_power_mw": opt_power / 1e6,
                "improvement_pct": improvement,
                "optimal_yaws_deg": opt_yaws.tolist(),
            })

            print(f"  WS={ws:5.1f}m/s, WD={wd:5.1f}°: "
                  f"greedy={greedy_power/1e6:.2f}MW -> opt={opt_power/1e6:.2f}MW "
                  f"({improvement:+.2f}%) yaw=[{opt_yaws.min():.1f}°, {opt_yaws.max():.1f}°]")

    avg_improvement = np.mean(improvements)
    print(f"\n{'='*60}")
    print(f"Average improvement over greedy: {avg_improvement:+.2f}%")
    print(f"Max improvement: {max(improvements):+.2f}%")
    print(f"Min improvement: {min(improvements):+.2f}%")
    print(f"{'='*60}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "avg_improvement_pct": avg_improvement,
        "max_improvement_pct": max(improvements),
        "conditions": results,
    }
    with open(results_dir / "optimizer_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_dir / 'optimizer_results.json'}")


if __name__ == "__main__":
    main()
