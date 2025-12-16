from __future__ import annotations
"""Task 2 standing controller using QP-based contact wrench allocation."""
import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from sim_runner import TORQUE_LIMITS, run_simulation, foot_contacts, get_viewer, get_trunk_state
import visualization_utils as viz
import debug_tools

# QP solver is now in qp_solver.py
from qp_solver import qp_controller, QPControllerResult

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class HeightProfile:
    hold_time: float = 1.0
    rise_time: float = 0.5
    drop_time: float = 1.0
    start_height: float = 0.45
    peak_height: float = 0.55
    final_height: float = 0.4

    def desired_height(self, t: float) -> tuple[float, float]:
        # Hold at start_height for hold_time seconds
        if t < self.hold_time:
            return self.start_height, 0.0
        
        # Rise from start_height to peak_height in rise_time seconds
        t2 = t - self.hold_time
        if t2 < self.rise_time:
            frac = t2 / self.rise_time
            height = self.start_height + frac * (self.peak_height - self.start_height)
            vel = (self.peak_height - self.start_height) / self.rise_time
            return height, vel
        
        # Drop from peak_height to final_height in drop_time seconds
        t3 = t2 - self.rise_time
        if t3 < self.drop_time:
            frac = t3 / self.drop_time
            height = self.peak_height + frac * (self.final_height - self.peak_height)
            vel = (self.final_height - self.peak_height) / self.drop_time
            return height, vel
        
        # Hold at final_height
        return self.final_height, 0.0



class StandingQPController:
    def __init__(self, profile: HeightProfile, debug_frames: int = 0):
        self.profile = profile
        self._debug_always = True  # Always print debug info for every QP solve
        self._prev_tau: Optional[np.ndarray] = None
        self._filter_alpha = 0.1  # Low-pass filter coefficient (lower = more smoothing)
        self._contact_history: list[bool] = []

        # Logging buffers
        self.log_time: list[float] = []
        self.log_root_pos: list[np.ndarray] = []
        self.log_root_vel: list[np.ndarray] = []
        self.log_joint_pos: list[np.ndarray] = []
        self.log_joint_vel: list[np.ndarray] = []
        self.log_tau_cmd: list[np.ndarray] = []
        self.log_tau_raw: list[np.ndarray] = []
        self.log_tau_contact: list[np.ndarray] = []
        self.log_posture_boost: list[np.ndarray] = []
        self.log_contact_forces: list[np.ndarray] = []
        self.log_height_des: list[np.ndarray] = []
        self.log_contact_flag: list[bool] = []
        
        # Video recording
        self.video_frames: list[np.ndarray] = []
        self.renderer: Optional[mujoco.Renderer] = None
        self.recording_enabled = False

    def enable_video_recording(self, model: mujoco.MjModel) -> None:
        """Enable video recording by initializing offscreen renderer."""
        try:
            self.renderer = viz.create_video_renderer(model)
            self.recording_enabled = True
            print(f"[INFO] Video recording enabled ({viz.VIDEO_WIDTH}x{viz.VIDEO_HEIGHT} @ {viz.VIDEO_FPS} fps)")
        except Exception as e:
            print(f"[WARN] Failed to initialize video renderer: {e}")
            self.renderer = None
            self.recording_enabled = False

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, t: float):

        
        # Desired world pose (standing at origin, upright)
        z_des, zd_des = self.profile.desired_height(t)
        pos_des = np.array([0.0, 0.0, z_des])  # [x, y, z] (MuJoCo world: x, y, z)
        vel_des = np.array([0.0, 0.0, zd_des])
        theta_des = 0.0  # Upright orientation
        # q_des = [-1.2471975512, 1.0707963268, -0.2, 1.0707963268]  # desired joint positions
        q_des = [-1.2471975512, 1.0707963268, -0.2, 1.0707963268] 
        qd_des = np.zeros(4)  # desired joint velocities
        # Call QP controller to compute contact forces and torques
        qp_result: QPControllerResult = qp_controller(
            model=model,
            data=data,
            pos_des=pos_des,
            vel_des=vel_des,
            theta_des=theta_des,
            q_des=q_des,
            qd_des=qd_des,
            left_enable=True,
            right_enable=True
        )
        
        # Extract results from QP controller
        # tau_contact = qp_result.tau_contact
        Fx_r, Fz_r, Fx_l, Fz_l = qp_result.contact_forces

        # Logging for plots
        
        trunk_state = get_trunk_state(model, data, "body_frame")
        x = trunk_state.pos[0]
        z = trunk_state.pos[2]
        theta = trunk_state.theta
        xd = trunk_state.vel[0]
        zd = trunk_state.vel[2]
        thetad = trunk_state.angvel[1]
        # Joint positions and velocities (assume actuated joints are 3:7)
        q = data.qpos[3:7].copy()
        qd = data.qvel[3:7].copy()
        self.log_time.append(t)
        self.log_root_pos.append(np.array([x, z, theta]))
        self.log_root_vel.append(np.array([xd, zd, thetad]))
        self.log_joint_pos.append(q.copy())
        self.log_joint_vel.append(qd.copy())
        self.log_tau_cmd.append(qp_result.tau.copy())
        # self.log_tau_raw.append(tau_raw.copy())
        self.log_tau_contact.append(qp_result.tau_contact.copy())
        # self.log_posture_boost.append(posture_boost.copy())
        self.log_contact_forces.append(np.array([Fx_r, Fz_r, Fx_l, Fz_l]))
        self.log_height_des.append(np.array([z_des, zd_des]))
        # self.log_contact_flag.append(in_contact_any)

        # Capture video frame if recording enabled
        if self.recording_enabled and self.renderer is not None:
            # Sample at VIDEO_FPS rate
            dt = model.opt.timestep
            frame_period = 1.0 / viz.VIDEO_FPS
            num_logged = len(self.log_time)
            expected_frames = int(t / frame_period)
            if len(self.video_frames) < expected_frames:
                self.renderer.update_scene(data)
                frame = self.renderer.render()
                self.video_frames.append((frame * 255).astype(np.uint8))
        
        # Return forces for visualization
        return qp_result.tau, {
            "raw_tau": qp_result.tau_contact.copy(),
            "contact_forces": qp_result.contact_forces.copy(),
            # "right_contact": right_contact,
            # "left_contact": left_contact,
        }

    def save_plots(self, output_dir: Optional[str] = None) -> None:
        """Save diagnostic plots using visualization utility."""
        log_data = {
            "time": self.log_time,
            "root_pos": self.log_root_pos,
            "root_vel": self.log_root_vel,
            "joint_pos": self.log_joint_pos,
            "joint_vel": self.log_joint_vel,
            "tau_cmd": self.log_tau_cmd,
            "tau_raw": self.log_tau_raw,
            "tau_contact": self.log_tau_contact,
            "posture_boost": self.log_posture_boost,
            "contact_forces": self.log_contact_forces,
            "height_des": self.log_height_des,
            "contact_flag": self.log_contact_flag,
        }
        # viz.save_standing_controller_plots(log_data, output_dir)

    def save_video(self, output_dir: Optional[str] = None) -> None:
        """Save recorded video frames to MP4 file."""
        viz.save_video(self.video_frames, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 standing QP controller")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--sim-time", type=float, default=3.0)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument(
        "--ignore-violations",
        action="store_true",
        help="continue simulation even if constraints are violated",
    )
    parser.add_argument(
        "--debug-frames",
        type=int,
        default=0,
        help="number of control frames to emit debug logs for (default: 0)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="directory to save diagnostic plots (default: <repo>/plots)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="directory to save video (default: <repo>/videos)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="disable video recording (useful for faster runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = StandingQPController(HeightProfile(), debug_frames=args.debug_frames)
    
    # Enable video recording unless disabled
    if not args.no_video:
        # Need to load model to initialize renderer
        from sim_runner import XML_PATH
        temp_model = mujoco.MjModel.from_xml_path(XML_PATH)
        controller.enable_video_recording(temp_model)
    
    result = run_simulation(
        controller,
        sim_time=args.sim_time,
        interactive=args.interactive,
        perturb=args.perturb,
        description="Task 2 QP standing",
        stop_on_violation=False,
    )
    
    controller.save_plots(args.plots_dir)
    if not args.no_video:
        controller.save_video(args.video_dir)
    print(f"[INFO] Simulation complete for t={result.final_time:.2f}s")


if __name__ == "__main__":
    main()
