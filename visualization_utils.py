"""Utility functions for saving plots and videos from simulation data."""
from __future__ import annotations

import os
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_FPS = 60
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Constants duplicated from task2.py to avoid circular imports
TORQUE_LIMITS = [30, 60, 30, 60]  # Nm for [hip, knee, hip, knee]
FZ_MIN = 10.0
FZ_MAX = 250.0


def save_standing_controller_plots(
    log_data: dict[str, list],
    output_dir: Optional[str] = None,
) -> None:
    """
    Save diagnostic plots for standing controller.
    
    Args:
        log_data: Dictionary containing logged simulation data with keys:
            - time: list of timestamps
            - root_pos: list of [x, z, theta] arrays
            - root_vel: list of [xd, zd, thetad] arrays
            - joint_pos: list of joint position arrays
            - joint_vel: list of joint velocity arrays
            - tau_cmd: list of commanded torque arrays
            - tau_raw: list of raw (pre-clip) torque arrays
            - tau_contact: list of contact torque arrays
            - posture_boost: list of posture boost torque arrays
            - contact_forces: list of [Fx_r, Fy_r, Fx_l, Fy_l] arrays
            - height_des: list of [z_des, zd_des] arrays
            - contact_flag: list of boolean contact flags
        output_dir: Directory to save plots (default: plots/)
    """
    if not log_data.get("time"):
        print("[INFO] No logged data to plot.")
        return

    out_dir = output_dir or os.path.join(HERE, "plots")
    os.makedirs(out_dir, exist_ok=True)

    time_arr = np.asarray(log_data["time"])
    root_pos = np.vstack(log_data["root_pos"])
    root_vel = np.vstack(log_data["root_vel"])
    joint_pos = np.vstack(log_data["joint_pos"])
    joint_vel = np.vstack(log_data["joint_vel"])
    tau_cmd = np.vstack(log_data["tau_cmd"])
    tau_raw = np.vstack(log_data["tau_raw"])
    tau_contact = np.vstack(log_data["tau_contact"])
    posture_boost = np.vstack(log_data["posture_boost"])
    contact_forces = np.vstack(log_data["contact_forces"])
    height_des = np.vstack(log_data["height_des"])
    contact_flag = np.asarray(log_data["contact_flag"], dtype=float)

    # State tracking figure
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs1[0].plot(time_arr, root_pos[:, 1], label="z (root)")
    axs1[0].plot(time_arr, height_des[:, 0], "--", label="z_des")
    axs1[0].set_ylabel("Height [m]")
    axs1[0].grid(True)
    axs1[0].legend(loc="upper right")

    axs1[1].plot(time_arr, root_pos[:, 0], label="x")
    axs1[1].plot(time_arr, root_pos[:, 2], label="theta")
    axs1[1].set_ylabel("Root pose [m, rad]")
    axs1[1].grid(True)
    axs1[1].legend(loc="upper right")

    axs1[2].plot(time_arr, height_des[:, 1], label="zd_des")
    axs1[2].plot(time_arr, root_vel[:, 1], label="zd")
    axs1[2].set_ylabel("Vertical vel [m/s]")
    axs1[2].set_xlabel("Time [s]")
    axs1[2].grid(True)
    axs1[2].legend(loc="upper right")

    fig1.tight_layout()
    fig1_path = os.path.join(out_dir, "task2_state_tracking.png")
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    # Joint angles and velocities
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for idx in range(4):
        axs2[0].plot(time_arr, joint_pos[:, idx], label=f"q{idx+1}")
    axs2[0].set_ylabel("Joint angle [rad]")
    axs2[0].grid(True)
    axs2[0].legend(loc="upper right")

    for idx in range(4):
        axs2[1].plot(time_arr, joint_vel[:, idx], label=f"q{idx+1}_dot")
    axs2[1].set_ylabel("Joint vel [rad/s]")
    axs2[1].set_xlabel("Time [s]")
    axs2[1].grid(True)
    axs2[1].legend(loc="upper right")

    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, "task2_joint_states.png")
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    # Torque profiles
    fig3, axs3 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for idx in range(4):
        axs3[idx].plot(time_arr, tau_cmd[:, idx], label="applied", linewidth=1.8)
        axs3[idx].plot(time_arr, tau_raw[:, idx], "--", label="pre-clip", linewidth=1.0)
        axs3[idx].plot(time_arr, tau_contact[:, idx], ":", label="contact", linewidth=1.0)
        axs3[idx].plot(time_arr, posture_boost[:, idx], label="posture boost", linewidth=0.9)
        axs3[idx].axhline(TORQUE_LIMITS[idx], color="k", linestyle="--", linewidth=0.5)
        axs3[idx].axhline(-TORQUE_LIMITS[idx], color="k", linestyle="--", linewidth=0.5)
        axs3[idx].set_ylabel(f"tau{idx+1} [Nm]")
        axs3[idx].grid(True)
        if idx == 0:
            axs3[idx].legend(loc="upper right")
    axs3[-1].set_xlabel("Time [s]")
    fig3.tight_layout()
    fig3_path = os.path.join(out_dir, "task2_torques.png")
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)

    # Contact forces and status
    fig4, axs4 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs4[0].plot(time_arr, contact_forces[:, 0], label="Fx_r")
    axs4[0].plot(time_arr, contact_forces[:, 2], label="Fx_l")
    axs4[0].set_ylabel("Fx [N]")
    axs4[0].grid(True)
    axs4[0].legend(loc="upper right")

    axs4[1].plot(time_arr, contact_forces[:, 1], label="Fy_r")
    axs4[1].plot(time_arr, contact_forces[:, 3], label="Fy_l")
    axs4[1].axhline(FZ_MIN, color="k", linestyle="--", linewidth=0.5, label="Fy bounds")
    axs4[1].axhline(FZ_MAX, color="k", linestyle="--", linewidth=0.5)
    axs4[1].set_ylabel("Fy [N]")
    axs4[1].grid(True)
    axs4[1].legend(loc="upper right")

    axs4[2].step(time_arr, contact_flag, where="post", label="both feet contact")
    axs4[2].set_ylabel("Contact")
    axs4[2].set_xlabel("Time [s]")
    axs4[2].set_yticks([0.0, 1.0], labels=["False", "True"])
    axs4[2].grid(True)
    axs4[2].legend(loc="upper right")

    fig4.tight_layout()
    fig4_path = os.path.join(out_dir, "task2_contact_forces.png")
    fig4.savefig(fig4_path, dpi=200)
    plt.close(fig4)

    print(f"[INFO] Saved Task 2 diagnostic plots to: {out_dir}")


def save_video(
    video_frames: list[np.ndarray],
    output_dir: Optional[str] = None,
    filename: str = "task2_qp_standing.mp4",
    fps: int = VIDEO_FPS,
) -> None:
    """
    Save recorded video frames to MP4 file.
    
    Args:
        video_frames: List of video frame arrays (height, width, 3)
        output_dir: Directory to save video (default: videos/)
        filename: Output filename (default: task2_qp_standing.mp4)
        fps: Frames per second (default: VIDEO_FPS)
    """
    if not video_frames:
        print("[INFO] No video frames to save.")
        return

    out_dir = output_dir or os.path.join(HERE, "videos")
    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(out_dir, filename)
    print(f"[INFO] Saving video ({len(video_frames)} frames at {fps} fps)...")
    imageio.mimwrite(video_path, video_frames, fps=fps)
    print(f"[INFO] Video saved to: {video_path}")


def create_video_renderer(
    model: mujoco.MjModel,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> mujoco.Renderer:
    """
    Create an offscreen renderer for video capture.
    
    Args:
        model: MuJoCo model
        width: Video width in pixels
        height: Video height in pixels
    
    Returns:
        MuJoCo Renderer instance
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    return renderer
