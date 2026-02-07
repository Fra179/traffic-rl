#!/usr/bin/env python3
"""Run SUMO-RL on the grid3x3 scenario and plot intersection traffic as a heatmap.

This is intentionally a small, self-contained demo script.

Examples:
  python experiments/sumo_rl_grid3x3_heatmap.py
  python experiments/sumo_rl_grid3x3_heatmap.py --gui --seconds 1200
  python experiments/sumo_rl_grid3x3_heatmap.py --metric density --seconds 600
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

from sumo_rl import SumoEnvironment


DEFAULT_SCENARIO_DIR = Path("/home/andrea/traffic-rl/scenarios/grid3x3")


def _resolve_scenario_dir(user_value: str | None) -> Path:
    if user_value is None:
        if DEFAULT_SCENARIO_DIR.exists():
            return DEFAULT_SCENARIO_DIR
        # Workspace-relative fallback (useful if the repo is cloned elsewhere)
        return (Path(__file__).resolve().parents[1] / "scenarios" / "grid3x3").resolve()
    return Path(user_value).expanduser().resolve()


def _sumo_binary_exists(use_gui: bool) -> bool:
    # SUMO-RL typically launches SUMO via subprocess; ensure the expected binary is reachable.
    binary = "sumo-gui" if use_gui else "sumo"
    return shutil.which(binary) is not None


def _load_tls_positions_from_net(net_file: Path) -> dict[str, tuple[float, float]]:
    """Parse the SUMO net XML and return {tls_id: (x, y)} for traffic-light junctions."""
    tree = ET.parse(net_file)
    root = tree.getroot()

    positions: dict[str, tuple[float, float]] = {}
    for junction in root.findall("junction"):
        if junction.get("type") != "traffic_light":
            continue
        junction_id = junction.get("id")
        x = junction.get("x")
        y = junction.get("y")
        if junction_id is None or x is None or y is None:
            continue
        positions[junction_id] = (float(x), float(y))
    return positions


def _grid_heatmap(
    ts_values: dict[str, float],
    ts_positions: dict[str, tuple[float, float]],
    title: str,
    output_png: Path,
    show: bool,
) -> None:
    # Keep only signals where we know positions.
    items = [(ts_id, ts_values[ts_id], ts_positions[ts_id]) for ts_id in ts_values.keys() if ts_id in ts_positions]
    if not items:
        raise RuntimeError("No traffic-signal positions found in net file.")

    xs = sorted({round(pos[0], 3) for _, _, pos in items})
    ys = sorted({round(pos[1], 3) for _, _, pos in items})
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}

    grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
    labels: dict[tuple[int, int], str] = {}

    for ts_id, value, (x, y) in items:
        xi = x_index[round(x, 3)]
        yi = y_index[round(y, 3)]
        grid[yi, xi] = value
        labels[(yi, xi)] = ts_id

    masked = np.ma.masked_invalid(grid)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(masked, origin="lower", interpolation="nearest", aspect="equal")

    ax.set_title(title)
    ax.set_xlabel("x (SUMO coords)")
    ax.set_ylabel("y (SUMO coords)")
    ax.set_xticks(range(len(xs)), [str(int(x)) if float(x).is_integer() else str(x) for x in xs])
    ax.set_yticks(range(len(ys)), [str(int(y)) if float(y).is_integer() else str(y) for y in ys])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Traffic intensity")

    # Annotate only the intersections that exist.
    for (yi, xi), ts_id in labels.items():
        v = grid[yi, xi]
        if np.isnan(v):
            continue
        ax.text(
            xi,
            yi,
            f"{ts_id}\n{v:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if v > np.nanmax(grid) * 0.6 else "black",
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="SUMO-RL grid3x3 demo: run and plot a traffic heatmap at intersections")
    parser.add_argument(
        "--scenario-dir",
        default=None,
        help="Path to the scenario folder (defaults to /home/andrea/traffic-rl/scenarios/grid3x3)",
    )
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI (requires sumo-gui)")
    parser.add_argument("--seconds", type=int, default=600, help="Simulation horizon in seconds")
    parser.add_argument("--delta-time", type=int, default=5, help="Seconds between actions (SUMO-RL delta_time)")
    parser.add_argument(
        "--metric",
        choices=["queued", "density", "wait"],
        default="queued",
        help="Traffic metric per intersection (aggregated over time)",
    )
    parser.add_argument(
        "--output",
        default=str((Path(__file__).resolve().parent / "grid3x3_heatmap.png").resolve()),
        help="Output PNG path",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window (in addition to saving)")
    parser.add_argument("--seed", type=int, default=0, help="SUMO seed (0 means deterministic-ish; change to vary)")
    args = parser.parse_args()

    scenario_dir = _resolve_scenario_dir(args.scenario_dir)
    net_file = scenario_dir / "grid3x3.net.xml"
    route_file = scenario_dir / "grid3x3.rou.xml"

    if not net_file.exists() or not route_file.exists():
        print(f"Scenario files not found in: {scenario_dir}", file=sys.stderr)
        print(f"Expected: {net_file} and {route_file}", file=sys.stderr)
        return 2

    if not _sumo_binary_exists(args.gui):
        binary = "sumo-gui" if args.gui else "sumo"
        print(f"Could not find '{binary}' in PATH.", file=sys.stderr)
        print("Install SUMO and/or export PATH so 'sumo'/'sumo-gui' are available.", file=sys.stderr)
        print("If you have SUMO installed, a common fix is to set SUMO_HOME and add $SUMO_HOME/bin to PATH.", file=sys.stderr)
        return 2

    ts_positions = _load_tls_positions_from_net(net_file)

    env = SumoEnvironment(
        net_file=str(net_file),
        route_file=str(route_file),
        use_gui=args.gui,
        num_seconds=int(args.seconds),
        delta_time=int(args.delta_time),
        single_agent=False,
        sumo_seed=int(args.seed),
        out_csv_name=None,
        sumo_warnings=True,
    )

    try:
        env.reset(seed=args.seed)

        per_ts_values: dict[str, list[float]] = {ts_id: [] for ts_id in env.ts_ids}

        dones = {"__all__": False}
        while not dones.get("__all__", False):
            actions = {ts_id: env.action_spaces(ts_id).sample() for ts_id in env.ts_ids}
            observations, rewards, dones, info = env.step(actions)

            for ts_id, ts in env.traffic_signals.items():
                if args.metric == "queued":
                    per_ts_values[ts_id].append(float(ts.get_total_queued()))
                elif args.metric == "density":
                    per_ts_values[ts_id].append(float(np.mean(ts.get_lanes_density())))
                elif args.metric == "wait":
                    per_ts_values[ts_id].append(float(np.mean(ts.get_accumulated_waiting_time_per_lane())))

        ts_summary = {ts_id: (float(np.mean(v)) if len(v) else 0.0) for ts_id, v in per_ts_values.items()}

    finally:
        env.close()

    metric_title = {
        "queued": "Mean queued vehicles",
        "density": "Mean lane density (0..1)",
        "wait": "Mean accumulated waiting time per lane",
    }[args.metric]

    output_png = Path(args.output).expanduser().resolve()
    _grid_heatmap(
        ts_values=ts_summary,
        ts_positions=ts_positions,
        title=f"grid3x3 – {metric_title}",
        output_png=output_png,
        show=args.show,
    )

    print(f"Saved heatmap to: {output_png}")
    print(f"Traffic lights plotted: {', '.join(sorted(ts_summary.keys()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
