
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
HIP_ANGLE_LIMIT = (-120.0 * RAD, 30.0 * RAD)
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
    raw_tau = tau_cmd
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
        # cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_follow")
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_frame")
        # viewer.cam.fixedcamid = cam_id
        if description:
            print(f"Viewer launched ({description}). Close window to stop.")

    stop_reason = None
    first_violation: Optional[str] = None
    step_count = 0
    prev_tau_cmd = np.zeros(model.nu, dtype=float)

    # --- Begin: Record q, qd, tau for plotting ---
    q_hist = []
    qd_hist = []
    tau_hist = []
    t_hist = []
    # --- End: Record q, qd, tau for plotting ---

    # --- Begin: Video recording setup ---
    import visualization_utils
    video_frames = []
    renderer = visualization_utils.create_video_renderer(model)
    VIDEO_FPS = getattr(visualization_utils, 'VIDEO_FPS', 60)
    video_frame_period = 1.0 / VIDEO_FPS
    next_video_time = 0.0
    # --- End: Video recording setup ---

    for step_idx in range(n_steps):
        step_start = time.time()
        # Record q, qd, tau, t
        q_hist.append(data.qpos[3:7].copy())
        qd_hist.append(data.qvel[3:7].copy())
        t_hist.append(data.time)
        ctrl_output = controller(model, data, data.time)
        # print("Control Output at time {:.2f}s: {}".format(data.time, ctrl_output))
        if isinstance(ctrl_output, tuple):
            tau_cmd, info = ctrl_output
        else:
            tau_cmd, info = ctrl_output, None
        tau_hist.append(np.copy(tau_cmd) if tau_cmd is not None else np.zeros(model.nu))

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

        # --- Begin: Video frame capture (at VIDEO_FPS only) ---
        if data.time >= next_video_time:
            # Manually set the camera to track the robot by updating camera parameters each frame
            renderer.update_scene(data, camera="side_follow")
            frame = renderer.render()
            video_frames.append(visualization_utils.frame_to_uint8_rgb(frame))
            next_video_time += video_frame_period
        # --- End: Video frame capture ---

        violation = None
        if enforce_default_constraints:
            print(f"[DEBUG][sim_runner] Current q and qd = {data.qpos[3:7]}, {data.qvel[3:7]}")
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


    # --- Begin: Save video ---
    # Determine task name and output directory (reuse logic below)
    import inspect
    import os
    caller_frame = inspect.stack()[1]
    caller_filename = os.path.basename(caller_frame.filename)
    task_name = None
    if caller_filename.startswith("task") and caller_filename.endswith(".py"):
        task_name = caller_filename[:-3]
    else:
        task_name = "sim_result"
    video_dir = os.path.join(HERE, f"{task_name}_result", "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_filename = f"{task_name}_sim.mp4"
    visualization_utils.save_video(video_frames, video_dir, video_filename)
    # --- End: Save video ---

    # --- Begin: Plot and save q, qd, tau over time ---
    import matplotlib.pyplot as plt
    q_hist = np.array(q_hist)
    qd_hist = np.array(qd_hist)
    tau_hist = np.array(tau_hist)
    t_hist = np.array(t_hist)

    # Determine task name and output directory
    import inspect
    import os
    caller_frame = inspect.stack()[1]
    caller_filename = os.path.basename(caller_frame.filename)
    task_name = None
    if caller_filename.startswith("task") and caller_filename.endswith(".py"):
        task_name = caller_filename[:-3]
    else:
        task_name = "sim_result"
    plot_dir = os.path.join(HERE, f"{task_name}_result", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path_q = os.path.join(plot_dir, "q_qd_vs_time.png")
    plot_path_tau = os.path.join(plot_dir, "tau_vs_time.png")



    # --- Begin: Separate plots for hip and knee joints (q, qd, tau) ---
    deg = 180.0 / np.pi
    # Hip joints: indices 0 (left), 2 (right)
    hip_indices = [0, 2]
    hip_names = ["Left Hip", "Right Hip"]
    fig_hip, axs_hip = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for idx, joint_idx in enumerate(hip_indices):
        axs_hip[0].plot(t_hist, q_hist[:, joint_idx] * deg, label=hip_names[idx])
        axs_hip[1].plot(t_hist, qd_hist[:, joint_idx], label=hip_names[idx])
        axs_hip[2].plot(t_hist, tau_hist[:, joint_idx], label=hip_names[idx])
    # Hip constraints
    axs_hip[0].axhline(-120, color='r', linestyle='--', linewidth=1, label='Angle Limit')
    axs_hip[0].axhline(30, color='r', linestyle='--', linewidth=1)
    axs_hip[1].axhline(30, color='r', linestyle='--', linewidth=1, label='Velocity Limit')
    axs_hip[1].axhline(-30, color='r', linestyle='--', linewidth=1)
    axs_hip[2].axhline(30, color='r', linestyle='--', linewidth=1, label='Torque Limit')
    axs_hip[2].axhline(-30, color='r', linestyle='--', linewidth=1)
    axs_hip[0].set_ylabel("Hip Angle (deg)")
    axs_hip[1].set_ylabel("Hip Velocity (rad/s)")
    axs_hip[2].set_ylabel("Hip Torque (Nm)")
    axs_hip[2].set_xlabel("Time (s)")
    axs_hip[0].set_title(f"Hip Joint Angles vs Time ({task_name})")
    axs_hip[1].set_title(f"Hip Joint Velocities vs Time ({task_name})")
    axs_hip[2].set_title(f"Hip Joint Torques vs Time ({task_name})")
    for ax in axs_hip:
        ax.legend()
    plt.tight_layout()
    hip_path = os.path.join(plot_dir, "hip_q_qd_tau_vs_time.png")
    plt.savefig(hip_path)
    plt.close(fig_hip)
    print(f"Saved hip joint plot to {hip_path}")

    # Knee joints: indices 1 (left), 3 (right)
    knee_indices = [1, 3]
    knee_names = ["Left Knee", "Right Knee"]
    fig_knee, axs_knee = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for idx, joint_idx in enumerate(knee_indices):
        axs_knee[0].plot(t_hist, q_hist[:, joint_idx] * deg, label=knee_names[idx])
        axs_knee[1].plot(t_hist, qd_hist[:, joint_idx], label=knee_names[idx])
        axs_knee[2].plot(t_hist, tau_hist[:, joint_idx], label=knee_names[idx])
    # Knee constraints
    axs_knee[0].axhline(0, color='b', linestyle='--', linewidth=1, label='Angle Limit')
    axs_knee[0].axhline(160, color='b', linestyle='--', linewidth=1)
    axs_knee[1].axhline(15, color='b', linestyle='--', linewidth=1, label='Velocity Limit')
    axs_knee[1].axhline(-15, color='b', linestyle='--', linewidth=1)
    axs_knee[2].axhline(60, color='b', linestyle='--', linewidth=1, label='Torque Limit')
    axs_knee[2].axhline(-60, color='b', linestyle='--', linewidth=1)
    axs_knee[0].set_ylabel("Knee Angle (deg)")
    axs_knee[1].set_ylabel("Knee Velocity (rad/s)")
    axs_knee[2].set_ylabel("Knee Torque (Nm)")
    axs_knee[2].set_xlabel("Time (s)")
    axs_knee[0].set_title(f"Knee Joint Angles vs Time ({task_name})")
    axs_knee[1].set_title(f"Knee Joint Velocities vs Time ({task_name})")
    axs_knee[2].set_title(f"Knee Joint Torques vs Time ({task_name})")
    for ax in axs_knee:
        ax.legend()
    plt.tight_layout()
    knee_path = os.path.join(plot_dir, "knee_q_qd_tau_vs_time.png")
    plt.savefig(knee_path)
    plt.close(fig_knee)
    print(f"Saved knee joint plot to {knee_path}")
    # --- End: Separate plots for hip and knee joints (q, qd, tau) ---

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