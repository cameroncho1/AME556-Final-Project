from __future__ import annotations
"""Task 2 standing controller using QP-based contact wrench allocation."""
import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from sim_runner import TORQUE_LIMITS, run_simulation
from sim_runner import get_viewer
import visualization_utils as viz
import debug_tools
try:
    from cvxopt import matrix, solvers  # type: ignore

    solvers.options["show_progress"] = False
    HAS_CVXOPT = True
except Exception:
    HAS_CVXOPT = False

HERE = os.path.dirname(os.path.abspath(__file__))

MASS_TOTAL = 8 + 2 * 0.25  # 0.2 kg body + 4x0.25 kg legs
GRAVITY = 9.81
WEIGHT_FORCE = MASS_TOTAL * GRAVITY
MU = 0.5  # Changed from 0.5 to match homework
FZ_MIN = 60.0  # Changed from 0.3 * WEIGHT_FORCE to match homework
FZ_MAX = 600.0  # Changed from 3.0 * WEIGHT_FORCE to match homework
MG_PENALTY_WEIGHT = 1  # Changed from 25.0 to match homework (was 315x too large!)
DEBUG_QP_FORCES = False
VERT_AXIS = 2  # MuJoCo's +Z is vertical axis

Kp_root_xz = 100
Kd_root_xz = 100
Kp_root_theta = 60
Kd_root_theta = 600
Kp_joint = 100.0  # Reduced from 120.0 to reduce oscillations
Kd_joint = 100.5  # Reduced from 25.0 to reduce oscillations


def _fmt_debug(arr: np.ndarray) -> str:
    """Format numpy arrays for concise debug output."""
    return np.array2string(np.asarray(arr, dtype=float), precision=4, suppress_small=False)


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


def solve_contact_qp(
    G: np.ndarray,
    wrench_des: np.ndarray,
    contact_left: bool,
    contact_right: bool,
) -> np.ndarray:
    """Solve QP (or LS fallback) for contact forces, supporting 1- or 2-foot contact.

    Returns forces ordered [Fx_r, Fz_r, Fx_l, Fz_l].
    G: grasp matrix (3 x n_forces), wrench_des: (3,)
    """

    def _clip_force(Fx: float, Fz: float) -> tuple[float, float]:
        Fz_clipped = np.clip(Fz, FZ_MIN, FZ_MAX)
        Fx_clipped = np.clip(Fx, -MU * Fz_clipped, MU * Fz_clipped)
        return Fx_clipped, Fz_clipped
        
    if not contact_left and not contact_right:
        return np.zeros(4)

    # Left-only contact
    if contact_left and not contact_right:
        # Only left foot: G_left is (3,2), F = [Fx_l, Fz_l]
        G_left = G[:, 2:4]
        Q = G_left.T @ G_left + 1e-6 * np.eye(2)
        p = -G_left.T @ wrench_des
        if HAS_CVXOPT:
            try:
                P = matrix(2.0 * Q)
                q = matrix(2.0 * p)
                Gc = matrix(
                    np.array(
                        [
                            [1.0, -MU],
                            [-1.0, -MU],
                            [0.0, -1.0],
                            [0.0, 1.0],
                        ],
                        dtype=float,
                    )
                )
                h = matrix(np.array([0.0, 0.0, -FZ_MIN, FZ_MAX], dtype=float))
                sol = solvers.qp(P, q, Gc, h)
                fx_l, fz_l = np.array(sol["x"]).flatten()
            except Exception:
                fx_l, fz_l = np.linalg.lstsq(G_left, wrench_des, rcond=None)[0]
        else:
            fx_l, fz_l = np.linalg.lstsq(G_left, wrench_des, rcond=None)[0]
        fx_l, fz_l = _clip_force(fx_l, fz_l)
        return np.array([0.0, 0.0, fx_l, fz_l])

    # Right-only contact
    if contact_right and not contact_left:
        # Only right foot: G_right is (3,2), F = [Fx_r, Fz_r]
        G_right = G[:, 0:2]
        Q = G_right.T @ G_right + 1e-6 * np.eye(2)
        p = -G_right.T @ wrench_des
        if HAS_CVXOPT:
            try:
                P = matrix(2.0 * Q)
                q = matrix(2.0 * p)
                Gc = matrix(
                    np.array(
                        [
                            [1.0, -MU],
                            [-1.0, -MU],
                            [0.0, -1.0],
                            [0.0, 1.0],
                        ],
                        dtype=float,
                    )
                )
                h = matrix(np.array([0.0, 0.0, -FZ_MIN, FZ_MAX], dtype=float))
                sol = solvers.qp(P, q, Gc, h)
                fx_r, fz_r = np.array(sol["x"]).flatten()
            except Exception:
                fx_r, fz_r = np.linalg.lstsq(G_right, wrench_des, rcond=None)[0]
        else:
            fx_r, fz_r = np.linalg.lstsq(G_right, wrench_des, rcond=None)[0]
        fx_r, fz_r = _clip_force(fx_r, fz_r)
        return np.array([fx_r, fz_r, 0.0, 0.0])

    # Both feet in contact (default 4-force problem)
    Q = G.T @ G + 1e-6 * np.eye(4)
    p = -G.T @ wrench_des

    if HAS_CVXOPT:
        try:
            P = matrix(2.0 * Q)
            q = matrix(2.0 * p)
            Gc = matrix(
                np.array(
                    [
                        [1.0, -MU, 0.0, 0.0],
                        [-1.0, -MU, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -MU],
                        [0.0, 0.0, -1.0, -MU],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, -1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )
            )
            h = matrix(np.array([0.0, 0.0, 0.0, 0.0, -FZ_MIN, FZ_MAX, -FZ_MIN, FZ_MAX], dtype=float))
            sol = solvers.qp(P, q, Gc, h)
            F = np.array(sol["x"]).flatten()
            return F
        except Exception:
            pass

    # Least-squares fallback (both feet)
    print("[WARNING] QP solver failed, using least-squares fallback.")
    F_ls = np.linalg.lstsq(G, wrench_des, rcond=None)[0]
    # Pad with zeros if underactuated (e.g., only one foot in contact)
    if F_ls.shape[0] < 4:
        F_ls = np.pad(F_ls, (0, 4 - F_ls.shape[0]))
    Fx_r, Fz_r, Fx_l, Fz_l = F_ls[:4]
    Fx_r, Fz_r = _clip_force(Fx_r, Fz_r)
    Fx_l, Fz_l = _clip_force(Fx_l, Fz_l)
    return np.array([Fx_r, Fz_r, Fx_l, Fz_l])


class StandingQPController:
    def __init__(self, profile: HeightProfile, debug_frames: int = 0):
        self.profile = profile
        self._debug_always = True  # Always print debug info for every QP solve
        self._prev_tau: Optional[np.ndarray] = None
        self._filter_alpha = 0.3  # Low-pass filter coefficient (lower = more smoothing)
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
        # Track contact stability (per-foot)
        left_contact, right_contact = foot_contacts(model, data)
        in_contact_any = left_contact or right_contact
        self._contact_history.append(in_contact_any)
        if len(self._contact_history) > 10:
            self._contact_history.pop(0)
        debug_this_frame = self._debug_always and in_contact_any
        if debug_this_frame:
            print("\n[DEBUG][task2] ----- frame start -----")
            print(f"[DEBUG][task2] sim_time={t:.4f}s")
            print(f"[DEBUG][task2] contact_left={left_contact} contact_right={right_contact}")
            print(f"[DEBUG][task2] contact_history={self._contact_history[-5:]}")


        # --- NEW: Use world trunk pose for base control ---
        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_frame")
        trunk_pos = data.xpos[trunk_id]  # (3,) world position
        trunk_mat = data.xmat[trunk_id].reshape(3, 3)  # (3,3) world orientation
        trunk_vel = data.cvel[trunk_id][:3]  # world linear velocity
        trunk_angvel = data.cvel[trunk_id][3:]  # world angular velocity
        body_angvel = trunk_mat.T @ trunk_angvel  # body frame angular velocity
        print("[DEBUG][task2] trunk_angvel:", _fmt_debug(trunk_angvel))
        # Desired world pose (standing at origin, upright)
        z_des, zd_des = self.profile.desired_height(t)
        pos_des = np.array([0.0, z_des, 0.0])  # [x, z, y] (MuJoCo world: x, y, z)
        vel_des = np.array([0.0, zd_des, 0.0])
        # For orientation, keep upright (identity rotation)
        theta_des = 0.0
        # Only regulate pitch (about world Y)
        # Compute pitch from rotation matrix: theta = atan2(-R[2,0], R[0,0])
        pitch = math.atan2(-trunk_mat[2, 0], trunk_mat[0, 0])
        print("[DEBUG][task2] pitch calculation:", -trunk_mat[2, 0], trunk_mat[0, 0], pitch)
        pitch_rate = body_angvel[1]  # Approx pitch rate from angvel
        print("[DEBUG][task2] pitch_rate calculation:", body_angvel[1])
        # PD for world position (only z)
        fz_des = Kp_root_xz * (pos_des[1] - trunk_pos[1]) + Kd_root_xz * (vel_des[1] - trunk_vel[1])
        fx_des = Kp_root_xz * (pos_des[0] - trunk_pos[0]) + Kd_root_xz * (vel_des[0] - trunk_vel[0])
        # Visualize debug in MuJoCo viewer
        debug_tools.draw_arrow(
            get_viewer(),
            trunk_pos,
            trunk_pos + np.array([fx_des, 0.0, fz_des]) * 0.01,
            np.array([0.0, 1.0, 0.0, 0.6]),
        )
        # PD for pitch
        tau_theta = Kp_root_theta * (theta_des - pitch) + Kd_root_theta * (0.0 - pitch_rate)
        debug_tools.draw_arrow(
            get_viewer(),
            trunk_pos,
            trunk_pos + np.array([0.0, pitch * 0.01, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.6]),
        )
        if debug_this_frame:
            print(f"[DEBUG][task2] trunk_pos={_fmt_debug(trunk_pos)}")
            print(f"[DEBUG][task2] trunk_vel={_fmt_debug(trunk_vel)}")
            print(f"[DEBUG][task2] pitch={pitch:.4f} pitch_rate={pitch_rate:.4f}")
            print(f"[DEBUG][task2] fx_des={fx_des:.4f} fz_des={fz_des:.4f} tau_theta={tau_theta:.4f}")

        # Posture tracking for actuated joints (indices 3:7)
        q_des = [-0.3, 1.5, -0.7, 1.5]  # desired joint positions
        qd_des = np.zeros(4)  # desired joint velocities
        tau_posture = np.zeros(4)
        for i in range(4):
            q_curr = data.qpos[3 + i]
            qd_curr = data.qvel[3 + i]
            tau_posture[i] = (
                Kp_joint * (q_des[i] - q_curr) + Kd_joint * (qd_des[i] - qd_curr)
            )
        # Posture boost to help maintain upright pose
        tau_bias = data.qfrc_bias[3:7].copy()

        # Desired trunk wrench: [fx_des, fz_des, tau_theta]
        wrench_des = np.array([fx_des, fz_des, tau_theta])
        # For floating base: do NOT map to joint torques, pass wrench_des to QP
        # tau_trunk = J_trunk.T @ wrench_des  # (do not use)
        # tau_des = tau_trunk - tau_bias  # (do not use)

        # Compute foot Jacobians for QP
        jacp_r = np.zeros((3, model.nv))
        jacp_l = np.zeros((3, model.nv))
        jacr_tmp = np.zeros((3, model.nv))
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        mujoco.mj_jacBody(model, data, jacp_r, jacr_tmp, right_id)
        mujoco.mj_jacBody(model, data, jacp_l, jacr_tmp, left_id)
        Jr = np.vstack([jacp_r[0, 3:7], jacp_r[VERT_AXIS, 3:7]])
        Jl = np.vstack([jacp_l[0, 3:7], jacp_l[VERT_AXIS, 3:7]])
        # Build grasp matrix G: maps [Fx_r, Fz_r, Fx_l, Fz_l] to trunk wrench [fx, fz, tau_theta]
        # Each column: effect of each force on trunk wrench
        # For each foot: [Fx, Fz] at foot position, moment arm to trunk (pitch)
        # G = [Jr; Jl] for x, z, pitch
        # Jr, Jl: (2,4) each, but we want (3,4):
        #   Row 0: Jr[0,:], Jl[0,:] (Fx)
        #   Row 1: Jr[1,:], Jl[1,:] (Fz)
        #   Row 2: pitch effect (moment arm):
        # For this biped, pitch is about world Y, so use foot x offset * Fz - foot z offset * Fx
        # But for now, use a simple stacking:
        G = np.zeros((3, 4))
        G[0, 0] = 1.0  # Fx_r
        G[1, 1] = 1.0  # Fz_r
        G[0, 2] = 1.0  # Fx_l
        G[1, 3] = 1.0  # Fz_l
        # Pitch (about Y):
        # tau_theta = (x_foot - x_trunk) * Fz - (z_foot - z_trunk) * Fx for each foot
        # Get foot and trunk positions in world
        right_foot_pos = data.xpos[right_id].copy()
        left_foot_pos = data.xpos[left_id].copy()
        trunk_pos_ = trunk_pos.copy()
        # Right foot
        dx_r = right_foot_pos[0] - trunk_pos_[0]
        dz_r = right_foot_pos[2] - trunk_pos_[2]
        # Left foot
        dx_l = left_foot_pos[0] - trunk_pos_[0]
        dz_l = left_foot_pos[2] - trunk_pos_[2]
        G[2, 0] = -dz_r  # Fx_r moment arm
        G[2, 1] = dx_r   # Fz_r moment arm
        G[2, 2] = -dz_l  # Fx_l moment arm
        G[2, 3] = dx_l   # Fz_l moment arm

        if debug_this_frame:
            print(f"[DEBUG][task2] wrench_des={_fmt_debug(wrench_des)}")
            print(f"[DEBUG][task2] tau_bias={_fmt_debug(tau_bias)}")
            print(f"[DEBUG][task2] Jr=\n{_fmt_debug(Jr)}")
            print(f"[DEBUG][task2] Jl=\n{_fmt_debug(Jl)}")
            print(f"[DEBUG][task2] G (grasp matrix)=\n{_fmt_debug(G)}")

        F = solve_contact_qp(
            G,
            wrench_des,
            contact_left=left_contact,
            contact_right=right_contact,
        )
        if debug_this_frame:
            print(f"[DEBUG][task2] contact_forces={_fmt_debug(F)}")
            # print(f"[DEBUG][task2] f_vert_target={WEIGHT_FORCE + f_vert_des:.4f} (mg + f_vert_des)")
        Fx_r, Fz_r, Fx_l, Fz_l = F
        Fx_r = -Fx_r  # MuJoCo frame convention
        Fx_l = -Fx_l  # MuJoCo frame convention
        Fz_r = -Fz_r  # MuJoCo frame convention
        Fz_l = -Fz_l  # MuJoCo frame convention
        tau_contact = Jr.T @ np.array([Fx_r, Fz_r]) + Jl.T @ np.array([Fx_l, Fz_l])
        posture_boost = 0.15 * tau_posture  # Increased from 0.01 for better stability
        tau = tau_contact + posture_boost
        tau_raw = tau.copy()

        # Apply low-pass filter to smooth torque commands and reduce flickering
        if self._prev_tau is not None and in_contact_any:
            tau_filtered = self._filter_alpha * tau_raw + (1 - self._filter_alpha) * self._prev_tau
        else:
            tau_filtered = tau_raw

        self._prev_tau = tau_filtered.copy()
        # Clip to per-joint torque limits to avoid constraint violations
        tau = np.clip(tau_filtered, -TORQUE_LIMITS, TORQUE_LIMITS)

        # Debug draw tau
        joint_ids = [3, 4, 6, 7]
        for i in range(4):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 3 + i)
            # joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            # print(data.xpos)
            joint_pos = data.xpos[joint_ids[i]]
            print(f"[DEBUG][task2] joint {i} ({joint_name}) tau={tau[i]:.4f} id={joint_ids[i]}")
            debug_tools.draw_arrow(
                get_viewer(),
                joint_pos,
                joint_pos + np.array([0.0, tau[i] * 0.01, 0.0]),
                np.array([0.0, 0.4, 1.0, 0.6]),
            )
        # Logging for plots
        x = trunk_pos[0]
        z = trunk_pos[2]
        theta = pitch
        xd = trunk_vel[0]
        zd = trunk_vel[2]
        thetad = pitch_rate
        # Joint positions and velocities (assume actuated joints are 3:7)
        q = data.qpos[3:7].copy()
        qd = data.qvel[3:7].copy()
        self.log_time.append(t)
        self.log_root_pos.append(np.array([x, z, theta]))
        self.log_root_vel.append(np.array([xd, zd, thetad]))
        self.log_joint_pos.append(q.copy())
        self.log_joint_vel.append(qd.copy())
        self.log_tau_cmd.append(tau.copy())
        self.log_tau_raw.append(tau_raw.copy())
        self.log_tau_contact.append(tau_contact.copy())
        self.log_posture_boost.append(posture_boost.copy())
        self.log_contact_forces.append(np.array([Fx_r, Fz_r, Fx_l, Fz_l]))
        self.log_height_des.append(np.array([z_des, zd_des]))
        self.log_contact_flag.append(in_contact_any)

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

        if debug_this_frame:
            print(f"[DEBUG][task2] tau_contact={_fmt_debug(tau_contact)}")
            print(f"[DEBUG][task2] posture_boost={_fmt_debug(posture_boost)}")
            print(f"[DEBUG][task2] tau_raw={_fmt_debug(tau_raw)}")
            print(f"[DEBUG][task2] tau_filtered={_fmt_debug(tau_filtered)}")
            if self._prev_tau is not None:
                print(f"[DEBUG][task2] filter_effect={_fmt_debug(tau_filtered - tau_raw)}")
            print(f"[DEBUG][task2] tau_clipped={_fmt_debug(tau)}")
            print(f"[DEBUG][task2] filter_alpha={self._filter_alpha}")
            print("[DEBUG][task2] ----- frame end -----\n")

        if not in_contact_any:
            return np.zeros(4), {}
        
        # Return forces for visualization
        return tau, {
            "raw_tau": tau_raw,
            "contact_forces": [Fx_r, Fz_r, Fx_l, Fz_l],
            "right_contact": right_contact,
            "left_contact": left_contact,
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
        viz.save_standing_controller_plots(log_data, output_dir)

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
        stop_on_violation=not args.ignore_violations,
    )
    
    controller.save_plots(args.plots_dir)
    if not args.no_video:
        controller.save_video(args.video_dir)
    print(f"[INFO] Simulation complete for t={result.final_time:.2f}s")


if __name__ == "__main__":
    main()
