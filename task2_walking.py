from __future__ import annotations
"""Task 2 walking controller using MPC footstep planner and QP/PD control."""
import argparse
import os
from dataclasses import dataclass
from typing import Optional
import mujoco
import numpy as np
from sim_runner import TORQUE_LIMITS, run_simulation, foot_contacts, get_viewer, get_trunk_state
import visualization_utils_walking as viz
import visualization_utils
import debug_tools
from mpc_walking_controller import mpc_walking_controller, MPCFootstepPlanner

HERE = os.path.dirname(os.path.abspath(__file__))

@dataclass
class WalkingProfile:
    speed: float = 0.5  # m/s
    direction: int = 1  # 1 for forward, -1 for backward
    sim_time: float = 6.0
    step_length: float = 0.25
    step_time: float = 0.5
    horizon: int = 10

class WalkingMPCController:
    def __init__(self, profile: WalkingProfile, debug_frames: int = 0):
        self.profile = profile
        self._debug_always = True
        self._planner = MPCFootstepPlanner(
            step_time=profile.step_time,
            horizon=profile.horizon,
            step_length_nom=profile.step_length,
            v_des=profile.speed
        )
        self._prev_tau: Optional[np.ndarray] = None
        self._filter_alpha = 0.1
        self.log_time: list[float] = []
        self.log_root_pos: list[np.ndarray] = []
        self.log_root_vel: list[np.ndarray] = []
        self.log_joint_pos: list[np.ndarray] = []
        self.log_joint_vel: list[np.ndarray] = []
        self.log_tau_cmd: list[np.ndarray] = []
        self.log_contact_forces: list[np.ndarray] = []
        self.log_footsteps: list[np.ndarray] = []
        self.video_frames: list[np.ndarray] = []
        self.renderer: Optional[mujoco.Renderer] = None
        self.recording_enabled = False

    def enable_video_recording(self, model: mujoco.MjModel) -> None:
        try:
            self.renderer = visualization_utils.create_video_renderer(model)
            self.recording_enabled = True
            print(f"[INFO] Video recording enabled ({visualization_utils.VIDEO_WIDTH}x{visualization_utils.VIDEO_HEIGHT} @ {visualization_utils.VIDEO_FPS} fps)")
        except Exception as e:
            print(f"[WARN] Failed to initialize video renderer: {e}")
            self.renderer = None
            self.recording_enabled = False

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, t: float):
        # Desired world pose (walking at speed)
        x_des = self.profile.speed * t * self.profile.direction
        z_des = 0.45
        pos_des = np.array([x_des, 0.0, z_des])
        vel_des = np.array([self.profile.speed * self.profile.direction, 0.0, 0.0])
        theta_des = 0.0
        # q_des = [-1.2471975512, 1.0707963268, -0.2, 1.0707963268]
        q_des = np.array([-0.5, 1.0, -0.5, 1.0])
        qd_des = np.zeros(4)
        # Call MPC walking controller
        qp_result = mpc_walking_controller(
            model=model,
            data=data,
            planner=self._planner,
            t=t,
            pos_des=pos_des,
            vel_des=vel_des,
            theta_des=theta_des,
            q_des=q_des,
            qd_des=qd_des,
            direction=self.profile.direction
        )
        # Logging
        trunk_state = get_trunk_state(model, data, "body_frame")
        x = trunk_state.pos[0]
        z = trunk_state.pos[2]
        theta = trunk_state.theta
        xd = trunk_state.vel[0]
        zd = trunk_state.vel[2]
        thetad = trunk_state.angvel[1]
        q = data.qpos[3:7].copy()
        qd = data.qvel[3:7].copy()
        self.log_time.append(t)
        self.log_root_pos.append(np.array([x, z, theta]))
        self.log_root_vel.append(np.array([xd, zd, thetad]))
        self.log_joint_pos.append(q.copy())
        self.log_joint_vel.append(qd.copy())
        self.log_tau_cmd.append(qp_result.tau.copy())
        self.log_contact_forces.append(qp_result.contact_forces.copy())
        self.log_footsteps.append(np.array(self._planner.footsteps))
        # Video
        if self.recording_enabled and self.renderer is not None:
            dt = model.opt.timestep
            frame_period = 1.0 / visualization_utils.VIDEO_FPS
            expected_frames = int(t / frame_period)
            if len(self.video_frames) < expected_frames:
                self.renderer.update_scene(data)
                frame = self.renderer.render()
                self.video_frames.append((frame * 255).astype(np.uint8))
        return qp_result.tau, {}

    def save_plots(self, output_dir: Optional[str] = None) -> None:
        log_data = {
            "time": self.log_time,
            "root_pos": self.log_root_pos,
            "root_vel": self.log_root_vel,
            "joint_pos": self.log_joint_pos,
            "joint_vel": self.log_joint_vel,
            "tau_cmd": self.log_tau_cmd,
            "contact_forces": self.log_contact_forces,
            "footsteps": self.log_footsteps,
        }
        viz.save_walking_controller_plots(log_data, output_dir)

    def save_video(self, output_dir: Optional[str] = None) -> None:
        visualization_utils.save_video(self.video_frames, output_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 walking MPC controller")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--sim-time", type=float, default=6.0)
    parser.add_argument("--direction", type=int, default=1, help="1 for forward, -1 for backward")
    parser.add_argument("--plots-dir", type=str, default=None)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    profile = WalkingProfile(direction=args.direction, sim_time=args.sim_time)
    controller = WalkingMPCController(profile)
    if not args.no_video:
        from sim_runner import XML_PATH
        temp_model = mujoco.MjModel.from_xml_path(XML_PATH)
        controller.enable_video_recording(temp_model)
    result = run_simulation(
        controller,
        sim_time=args.sim_time,
        interactive=args.interactive,
        description="Task 2 MPC walking",
        stop_on_violation=False,
    )
    out_dir = args.plots_dir or os.path.join(HERE, "task2_walking_result", "plots")
    vid_dir = args.video_dir or os.path.join(HERE, "task2_walking_result", "videos")
    controller.save_plots(out_dir)
    if not args.no_video:
        controller.save_video(vid_dir)
    print(f"[INFO] Walking simulation complete for t={result.final_time:.2f}s")

if __name__ == "__main__":
    main()
