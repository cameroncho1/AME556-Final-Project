"""Task 1 controller entry point using shared sim runner."""
from __future__ import annotations

import argparse
import mujoco
import numpy as np

from sim_runner import run_simulation

HIP_TORQUE_LIMIT = 30.0
KNEE_TORQUE_LIMIT = 60.0
TORQUE_LIMITS = np.array([HIP_TORQUE_LIMIT, KNEE_TORQUE_LIMIT, HIP_TORQUE_LIMIT, KNEE_TORQUE_LIMIT])
Kp = np.array([50.0, 50.0, 50.0, 50.0])
Kd = np.array([2.0, 2.0, 2.0, 2.0])
Q_DES = np.array([-np.pi / 3, np.pi / 2, 0.0, np.pi / 2])


class Task1Controller:
    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, _: float):
        if self.mode == "zero":
            tau_cmd = np.zeros(4)
            info = {"raw_tau": tau_cmd.copy()}
            return tau_cmd, info

        q = data.qpos[3:7]
        qd = data.qvel[3:7]
        tau_raw = Kp * (Q_DES - q) - Kd * qd
        tau_sat = np.clip(tau_raw, -TORQUE_LIMITS, TORQUE_LIMITS)
        info = {"raw_tau": tau_raw.copy()}
        return tau_sat, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 constraint-aware simulation")
    parser.add_argument("--interactive", action="store_true", help="launch viewer")
    parser.add_argument("--controller", choices=["pd", "zero"], default="pd")
    parser.add_argument("--sim-time", type=float, default=3.0)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument(
        "--ignore-violations",
        action="store_true",
        help="continue simulation even if constraints are violated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = Task1Controller(args.controller)
    run_simulation(
        controller,
        sim_time=args.sim_time,
        interactive=args.interactive,
        perturb=args.perturb,
        description="Task 1",
        stop_on_violation=not args.ignore_violations,
    )


if __name__ == "__main__":
    main()
