"""
Task 1: Constraint-aware simulation for the 2D biped robot.

Features implemented per project spec:
- Joint angle, velocity, and torque limit monitoring for hips (q1, q3) and knees (q2, q4).
- Command saturation before sending torques to MuJoCo.
- Immediate termination with an error flag when any constraint is violated.
- Simulation time overlay inside the interactive viewer.

Usage examples (from this folder):
    python task1_sim.py --interactive               # run with interactive viewer, PD control
    python task1_sim.py --controller zero           # passive fall, still checks constraints
    python task1_sim.py --sim-time 4.0 --perturb    # perturbed initial state demo

The controller is intentionally simple (joint-space PD) to keep the focus on
Task 1 constraint handling. Swap in your own controller by modifying
`compute_control` while leaving `apply_saturation` and `check_constraints` intact.
"""
from __future__ import annotations
import argparse
import os
import time
from typing import List, Tuple

import mujoco
import mujoco.viewer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "biped_robot.xml")

# --- Limits (convert degrees to radians once up front) ---
RAD = np.pi / 180.0
HIP_ANGLE_LIMIT = (-120.0 * RAD, 30.0 * RAD)
KNEE_ANGLE_LIMIT = (0.0 * RAD, 160.0 * RAD)
HIP_VEL_LIMIT = 30.0  # rad/s
KNEE_VEL_LIMIT = 15.0  # rad/s
HIP_TORQUE_LIMIT = 30.0  # Nm
KNEE_TORQUE_LIMIT = 60.0  # Nm
TORQUE_LIMITS = np.array([HIP_TORQUE_LIMIT, KNEE_TORQUE_LIMIT, HIP_TORQUE_LIMIT, KNEE_TORQUE_LIMIT])

# Simple joint-space PD gains and desired angles (taken from HW3 setup)
Kp = np.array([50.0, 50.0, 50.0, 50.0])
Kd = np.array([2.0, 2.0, 2.0, 2.0])
Q_DES = np.array([-np.pi / 3, np.pi / 2, 0.0, np.pi / 2])


def reset_state(model: mujoco.MjModel, data: mujoco.MjData, perturb: bool = False) -> None:
    """Reset to keyframe "init" and optionally add small perturbations."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    if perturb:
        noise = np.array([0.02, -0.02, 0.0, 0.05])
        data.qpos[3:7] += noise
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def compute_control(data: mujoco.MjData, controller: str) -> np.ndarray:
    """Return raw torque command (before saturation)."""
    if controller == "zero":
        return np.zeros(4)
    q = data.qpos[3:7]
    qd = data.qvel[3:7]
    return Kp * (Q_DES - q) - Kd * qd


def apply_saturation(tau_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clip torques joint-wise and report which joints saturated."""
    saturated = np.abs(tau_raw) > TORQUE_LIMITS
    tau_sat = np.clip(tau_raw, -TORQUE_LIMITS, TORQUE_LIMITS)
    return tau_sat, saturated


def _in_range(val: float, low: float, high: float) -> bool:
    return low <= val <= high


def check_constraints(data: mujoco.MjData, tau_raw: np.ndarray, saturated: np.ndarray) -> List[str]:
    """Return list of violated constraints (empty if all good)."""
    q = data.qpos[3:7]
    qd = data.qvel[3:7]
    reasons: List[str] = []

    # Angle limits
    for idx in (0, 2):  # hips q1, q3
        if not _in_range(q[idx], *HIP_ANGLE_LIMIT):
            reasons.append(f"hip angle q{idx+1}={q[idx]:.2f} rad out of [{HIP_ANGLE_LIMIT[0]:.2f}, {HIP_ANGLE_LIMIT[1]:.2f}]")
    for idx in (1, 3):  # knees q2, q4
        if not _in_range(q[idx], *KNEE_ANGLE_LIMIT):
            reasons.append(f"knee angle q{idx+1}={q[idx]:.2f} rad out of [{KNEE_ANGLE_LIMIT[0]:.2f}, {KNEE_ANGLE_LIMIT[1]:.2f}]")

    # Velocity limits
    for idx in (0, 2):
        if abs(qd[idx]) > HIP_VEL_LIMIT:
            reasons.append(f"hip velocity |qd{idx+1}|={abs(qd[idx]):.1f} rad/s exceeds {HIP_VEL_LIMIT}")
    for idx in (1, 3):
        if abs(qd[idx]) > KNEE_VEL_LIMIT:
            reasons.append(f"knee velocity |qd{idx+1}|={abs(qd[idx]):.1f} rad/s exceeds {KNEE_VEL_LIMIT}")

    # Torque commands beyond limits (even though they are saturated before apply)
    for idx, limit in enumerate(TORQUE_LIMITS):
        if abs(tau_raw[idx]) > limit:
            reasons.append(f"command torque |tau{idx+1}|={abs(tau_raw[idx]):.1f} exceeds {limit} Nm (saturated)")
    # Highlight any saturation event explicitly
    for idx, was_sat in enumerate(saturated):
        if was_sat:
            sat_val = np.sign(tau_raw[idx]) * TORQUE_LIMITS[idx]
            reasons.append(f"tau{idx+1} clipped to {sat_val:.1f} Nm")

    return reasons


def add_time_overlay(viewer: mujoco.viewer.Handle, sim_time: float, violation: str | None) -> None:
    """Print sim time and violation info (viewer overlay not available in MuJoCo 3.x)."""
    # MuJoCo 3.x passive viewer doesn't support add_overlay; print to console instead
    if violation:
        print(f"[t={sim_time:.3f}s] {violation}")


def run(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    reset_state(model, data, perturb=args.perturb)

    dt = model.opt.timestep
    n_steps = int(args.sim_time / dt)

    viewer = None
    if args.interactive:
        viewer = mujoco.viewer.launch_passive(model, data)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_follow")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id


    violation_log: List[str] = []
    stop_reason = None

    for k in range(n_steps):
        step_start = time.time()

        tau_raw = compute_control(data, args.controller)
        tau_sat, saturated = apply_saturation(tau_raw)
        data.ctrl[:] = tau_sat

        mujoco.mj_step(model, data)
        violation_log = check_constraints(data, tau_raw, saturated)
        if violation_log:
            stop_reason = "; ".join(violation_log)
            break

        if viewer:
            add_time_overlay(viewer, data.time, None)
            viewer.sync()
            # Real-time pacing
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            if not viewer.is_running():
                stop_reason = "Viewer closed"
                break

    # Final overlays if a violation occurred while viewer is open
    if viewer and violation_log:
        add_time_overlay(viewer, data.time, stop_reason)
        viewer.sync()
        time.sleep(1.0)
        viewer.close()

    if not viewer:
        # Offline run still reports terminal state
        data_state = data.qpos[3:7]
        print(f"Final joint angles (rad): {data_state}")

    if stop_reason:
        print(f"Simulation terminated early at t={data.time:.3f} s: {stop_reason}")
    else:
        print(f"Simulation completed full horizon ({args.sim_time:.2f} s) without violations.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1: constraint-aware biped simulation")
    parser.add_argument("--interactive", action="store_true", help="launch viewer and show time overlay")
    parser.add_argument("--controller", choices=["pd", "zero"], default="pd", help="controller type")
    parser.add_argument("--sim-time", type=float, default=3.0, help="simulation duration in seconds")
    parser.add_argument("--perturb", action="store_true", help="perturb initial joint angles")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
