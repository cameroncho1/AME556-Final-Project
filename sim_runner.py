
from __future__ import annotations
"""Shared MuJoCo simulation utilities for AME556 final project tasks."""
_GLOBAL_VIEWER = None
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import mujoco
import mujoco.viewer
import numpy as np

# from debug_tools import draw_contact_force_arrows
import debug_tools
HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "biped_robot.xml")

ControllerFn = Callable[[mujoco.MjModel, mujoco.MjData, float], Any]
ConstraintFn = Callable[[mujoco.MjModel, mujoco.MjData, np.ndarray, Optional[dict[str, Any]]], Optional[str]]
ResetFn = Callable[[mujoco.MjModel, mujoco.MjData, bool], None]
RAD = np.pi / 180.0
HIP_ANGLE_LIMIT = (-90.0 * RAD, 90.0 * RAD)
KNEE_ANGLE_LIMIT = (0.0 * RAD, 160.0 * RAD)
HIP_VEL_LIMIT = 30.0  # rad/s
KNEE_VEL_LIMIT = 15.0  # rad/s
HIP_TORQUE_LIMIT = 30.0  # Nm
KNEE_TORQUE_LIMIT = 60.0  # Nm
TORQUE_LIMITS = np.array([HIP_TORQUE_LIMIT, KNEE_TORQUE_LIMIT, HIP_TORQUE_LIMIT, KNEE_TORQUE_LIMIT])
MU = 0.5  # Friction coefficient

# Interactive viewer camera is driven by the XML camera `side_follow`
# so that interactive view matches saved video.

def _in_range(val: float, low: float, high: float) -> bool:
    return low <= val <= high


def default_constraint_checker(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau_cmd: np.ndarray,
    info: Optional[dict[str, Any]],
) -> Optional[str]:
    """Joint/torque constraint checks shared by all tasks."""
    reasons = []
    q = data.qpos[3:7]
    qd = data.qvel[3:7]
    raw_tau = info.get("raw_tau") if (info and "raw_tau" in info) else tau_cmd

    for idx in (0, 2):
        if not _in_range(q[idx], *HIP_ANGLE_LIMIT):
            reasons.append(
                f"hip angle q{idx+1}={q[idx]:.2f} rad outside [{HIP_ANGLE_LIMIT[0]:.2f}, {HIP_ANGLE_LIMIT[1]:.2f}]"
            )
    for idx in (1, 3):
        if not _in_range(q[idx], *KNEE_ANGLE_LIMIT):
            reasons.append(
                f"knee angle q{idx+1}={q[idx]:.2f} rad outside [{KNEE_ANGLE_LIMIT[0]:.2f}, {KNEE_ANGLE_LIMIT[1]:.2f}]"
            )

    for idx in (0, 2):
        if abs(qd[idx]) > HIP_VEL_LIMIT:
            reasons.append(f"hip velocity |qd{idx+1}|={abs(qd[idx]):.1f} > {HIP_VEL_LIMIT}")
    for idx in (1, 3):
        if abs(qd[idx]) > KNEE_VEL_LIMIT:
            reasons.append(f"knee velocity |qd{idx+1}|={abs(qd[idx]):.1f} > {KNEE_VEL_LIMIT}")

    for idx, limit in enumerate(TORQUE_LIMITS):
        # raw_tau if provided, else actual applied control
        tau_val = raw_tau[idx] if raw_tau is not None else tau_cmd[idx]
        if abs(tau_val) > limit:
            reasons.append(f"command torque |tau{idx+1}|={abs(tau_val):.1f} exceeds {limit} Nm")

    return "; ".join(reasons) if reasons else None


@dataclass
class SimResult:
    final_time: float
    violation: Optional[str]
    steps: int


def default_reset(model: mujoco.MjModel, data: mujoco.MjData, perturb: bool = False) -> None:
    """Reset to the XML keyframe 'init' and optionally add joint perturbations."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    if perturb:
        noise = np.array([0.02, -0.02, 0.02, -0.02])
        data.qpos[3:7] += noise
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def run_simulation(
        # import debug_tools
    controller: ControllerFn,
    *,
    sim_time: float = 3.0,
    interactive: bool = False,
    perturb: bool = False,
    constraint_checker: Optional[ConstraintFn] = None,
    reset_fn: Optional[ResetFn] = None,
    description: str = "",
    enforce_default_constraints: bool = True,
    stop_on_violation: bool = True,
) -> SimResult:
    """Run MuJoCo simulation with a user-provided controller.

    The controller is called as ``controller(model, data, t)`` and can return either
    ``tau`` directly or ``(tau, info_dict)`` where ``info_dict`` carries auxiliary data
    (e.g., pre-saturation torques) for constraint checkers.
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    (reset_fn or default_reset)(model, data, perturb)

    dt = model.opt.timestep
    n_steps = int(1e9)  # Effectively infinite steps; run until viewer closed or violation
    global _GLOBAL_VIEWER
    viewer: Optional[mujoco.viewer.Handle] = None

    if interactive:
        viewer = mujoco.viewer.launch_passive(model, data)
        _GLOBAL_VIEWER = viewer
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_follow")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = cam_id
        if description:
            print(f"Viewer launched ({description}). Close window to stop.")

    stop_reason = None
    first_violation: Optional[str] = None
    step_count = 0
    prev_tau_cmd = np.zeros(model.nu, dtype=float)

    for step_idx in range(n_steps):
        step_start = time.time()
        ctrl_output = controller(model, data, data.time)
        # print("Control Output at time {:.2f}s: {}".format(data.time, ctrl_output))
        if isinstance(ctrl_output, tuple):
            tau_cmd, info = ctrl_output
        else:
            tau_cmd, info = ctrl_output, None

        if tau_cmd is None:
            tau_cmd = prev_tau_cmd.copy()
            print(f"[WARNING] Controller returned None tau at t={data.time:.3f} s; reusing previous command.")
        else:
            tau_cmd = np.asarray(tau_cmd, dtype=float)
            prev_tau_cmd = tau_cmd.copy()
        # print("Applied Torques at time {:.2f}s: {}".format(data.time, tau_cmd))
        data.ctrl[:] = tau_cmd
        mujoco.mj_step(model, data)
        step_count += 1

        violation = None
        if enforce_default_constraints:
            violation = default_constraint_checker(model, data, tau_cmd, info)
        if not violation and constraint_checker:
            violation = constraint_checker(model, data, tau_cmd, info)
        if violation:
            if stop_on_violation:
                stop_reason = violation
                break
            if first_violation is None:
                first_violation = f"t={data.time:.3f} s: {violation}"
            # print(f"[WARNING] Constraint violation ignored at t={data.time:.3f} s: {violation}")

        if viewer:
            # Clear debug arrows at the start of each frame
            # Draw contact force arrows if provided in info
            # if info and "contact_forces" in info:
            #     debug_tools.draw_contact_force_arrows(viewer, model, data, info)
            
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            if not viewer.is_running():
                stop_reason = "Viewer closed"
                break
            
            debug_tools.clear_arrows(viewer)

    if viewer:
        viewer.close()
        _GLOBAL_VIEWER = None

    final_violation = stop_reason or first_violation

    if stop_reason:
        print(f"Simulation stopped at t={data.time:.3f} s: {stop_reason}")
    else:
        print(f"Simulation completed {sim_time:.2f} s without reported violations.")
        if first_violation:
            print(f"Note: constraint violations occurred but simulation continued (first: {first_violation}).")

    return SimResult(final_time=data.time, violation=final_violation, steps=step_count)

def get_viewer():
    global _GLOBAL_VIEWER
    return _GLOBAL_VIEWER

def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool]:
    """Return (left_in_contact, right_in_contact) using body IDs."""
    floor_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground_body")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    right = left = False
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        if geom1 < 0 or geom2 < 0:
            continue
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]
        bodies = {body1, body2}
        if right_body in bodies and floor_body in bodies:
            right = True
        if left_body in bodies and floor_body in bodies:
            left = True
    return left, right


@dataclass
class TrunkState:
    """Trunk state: position (x, z), orientation (theta), and velocities."""
    x: float
    z: float
    theta: float  # pitch angle about world Y axis
    xd: float  # x velocity
    zd: float  # z velocity
    thetad: float  # pitch rate
    pos: np.ndarray  # (3,) world position
    vel: np.ndarray  # (3,) world linear velocity
    angvel: np.ndarray  # (3,) world angular velocity


def get_trunk_state(model: mujoco.MjModel, data: mujoco.MjData, body_name: str = "body_frame") -> TrunkState:
    """Extract trunk state (x, z, theta and velocities) from MuJoCo simulation.
    
    Returns TrunkState with position, orientation, and velocities.
    """
    import math
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    trunk_pos = data.xpos[trunk_id].copy()  # (3,) world position
    trunk_mat = data.xmat[trunk_id].reshape(3, 3)  # (3,3) world orientation
    trunk_vel = data.cvel[trunk_id][3:].copy()  # world linear velocity
    trunk_angvel = data.cvel[trunk_id][:3].copy()  # world angular velocity
    
    # Extract x, z, theta
    x = trunk_pos[0]
    z = trunk_pos[2]
    # Compute pitch from rotation matrix: theta = atan2(-R[2,0], R[0,0])
    theta = math.atan2(-trunk_mat[2, 0], trunk_mat[0, 0])
    
    # Extract velocities
    xd = trunk_vel[0]
    zd = trunk_vel[2]
    thetad = trunk_angvel[1]  # pitch rate (about Y axis)
    
    return TrunkState(
        x=x, z=z, theta=theta,
        xd=xd, zd=zd, thetad=thetad,
        pos=trunk_pos, vel=trunk_vel, angvel=trunk_angvel
    )