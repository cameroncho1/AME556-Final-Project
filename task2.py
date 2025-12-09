"""Task 2 standing controller using QP-based contact wrench allocation."""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from sim_runner import TORQUE_LIMITS, run_simulation
import visualization_utils as viz

try:
    from cvxopt import matrix, solvers  # type: ignore

    solvers.options["show_progress"] = False
    HAS_CVXOPT = True
except Exception:
    HAS_CVXOPT = False

HERE = os.path.dirname(os.path.abspath(__file__))

MASS_TOTAL = 8 + 4 * 0.25  # 0.2 kg body + 4x0.25 kg legs
GRAVITY = 9.81
WEIGHT_FORCE = MASS_TOTAL * GRAVITY
MU = 0.5  # Changed from 0.5 to match homework
FZ_MIN = 6.0  # Changed from 0.3 * WEIGHT_FORCE to match homework
FZ_MAX = 10.0  # Changed from 3.0 * WEIGHT_FORCE to match homework
MG_PENALTY_WEIGHT = 1  # Changed from 25.0 to match homework (was 315x too large!)
DEBUG_QP_FORCES = False
VERT_AXIS = 2  # MuJoCo's +Z is vertical axis

Kp_root_xz = 120
Kd_root_xz = 10
Kp_root_theta = 1200.0
Kd_root_theta = 10.0
Kp_joint = 120.0  # Reduced from 120.0 to reduce oscillations
Kd_joint = 10.5  # Reduced from 25.0 to reduce oscillations

# Walking parameters
STEP_LENGTH = 0.08  # meters of forward step target (heuristic)
STEP_HEIGHT = 0.04  # meters of swing foot lift (approximated via joint offsets)
STEP_KNEE_BEND = 0.35  # rad knee flex during swing
PHASE_DURATIONS = {
    "double1": 0.4,
    "left_stance": 0.4,
    "double2": 0.4,
    "right_stance": 0.4,
}


def phase_schedule(t: float) -> tuple[str, float, float]:
    """Return (phase_name, t_in_phase, phase_duration) for a simple scripted gait."""
    t_mod = t % sum(PHASE_DURATIONS.values())
    accum = 0.0
    for name, dur in PHASE_DURATIONS.items():
        if t_mod < accum + dur:
            return name, t_mod - accum, dur
        accum += dur
    # Fallback
    last_name = list(PHASE_DURATIONS.keys())[-1]
    return last_name, t_mod - accum, PHASE_DURATIONS[last_name]


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
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shin")
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shin")
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
    Jr: np.ndarray,
    Jl: np.ndarray,
    tau_des: np.ndarray,
    f_vert_des: float,
    contact_left: bool,
    contact_right: bool,
) -> np.ndarray:
    """Solve QP (or LS fallback) for contact forces, supporting 1- or 2-foot contact.

    Returns forces ordered [Fx_r, Fz_r, Fx_l, Fz_l]."""

    def _clip_force(Fx: float, Fz: float) -> tuple[float, float]:
        Fz_clipped = np.clip(Fz, FZ_MIN, FZ_MAX)
        Fx_clipped = np.clip(Fx, -MU * Fz_clipped, MU * Fz_clipped)
        return Fx_clipped, Fz_clipped

    mg = WEIGHT_FORCE
    v = np.array([0.0, 1.0, 0.0, 1.0])
    weight = MG_PENALTY_WEIGHT
    f_vert_target = mg + f_vert_des

    # No contact: zero forces
    if not contact_left and not contact_right:
        return np.zeros(4)

    # Left-only contact
    if contact_left and not contact_right:
        A = Jl.T  # (4x2)
        Q = A.T @ A + 1e-6 * np.eye(2)
        p = -A.T @ tau_des + weight * np.array([0.0, -f_vert_target])
        if HAS_CVXOPT:
            try:
                P = matrix(2.0 * Q)
                q = matrix(2.0 * p)
                G = matrix(
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
                sol = solvers.qp(P, q, G, h)
                fx_l, fz_l = np.array(sol["x"]).flatten()
            except Exception:
                fx_l, fz_l = np.linalg.lstsq(A, tau_des, rcond=None)[0]
        else:
            fx_l, fz_l = np.linalg.lstsq(A, tau_des, rcond=None)[0]
        fx_l, fz_l = _clip_force(fx_l, fz_l)
        return np.array([0.0, 0.0, fx_l, fz_l])

    # Right-only contact
    if contact_right and not contact_left:
        A = Jr.T  # (4x2)
        Q = A.T @ A + 1e-6 * np.eye(2)
        p = -A.T @ tau_des + weight * np.array([0.0, -f_vert_target])
        if HAS_CVXOPT:
            try:
                P = matrix(2.0 * Q)
                q = matrix(2.0 * p)
                G = matrix(
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
                sol = solvers.qp(P, q, G, h)
                fx_r, fz_r = np.array(sol["x"]).flatten()
            except Exception:
                fx_r, fz_r = np.linalg.lstsq(A, tau_des, rcond=None)[0]
        else:
            fx_r, fz_r = np.linalg.lstsq(A, tau_des, rcond=None)[0]
        fx_r, fz_r = _clip_force(fx_r, fz_r)
        return np.array([fx_r, fz_r, 0.0, 0.0])

    # Both feet in contact (default 4-force problem)
    A = np.zeros((4, 4))
    A[:, 0:2] = Jr.T
    A[:, 2:4] = Jl.T
    Q = A.T @ A + 1e-6 * np.eye(4)
    p = -A.T @ tau_des
    Q += weight * np.outer(v, v)
    p += -weight * f_vert_target * v

    if HAS_CVXOPT:
        try:
            P = matrix(2.0 * Q)
            q = matrix(2.0 * p)
            G_list = [
                [1.0, -MU, 0.0, 0.0],
                [-1.0, -MU, 0.0, 0.0],
                [0.0, 0.0, 1.0, -MU],
                [0.0, 0.0, -1.0, -MU],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            h_list = [0.0, 0.0, 0.0, 0.0, -FZ_MIN, FZ_MAX, -FZ_MIN, FZ_MAX]
            G = matrix(np.array(G_list, dtype=float))
            h = matrix(np.array(h_list, dtype=float))
            sol = solvers.qp(P, q, G, h)
            F = np.array(sol["x"]).flatten()
            return F
        except Exception:
            pass

    # Least-squares fallback (both feet)
    print("[WARNING] QP solver failed, using least-squares fallback.")
    F_ls = np.linalg.pinv(A) @ tau_des
    Fx_r, Fz_r, Fx_l, Fz_l = F_ls
    Fx_r, Fz_r = _clip_force(Fx_r, Fz_r)
    Fx_l, Fz_l = _clip_force(Fx_l, Fz_l)
    return np.array([Fx_r, Fz_r, Fx_l, Fz_l])


class StandingQPController:
    def __init__(self, profile: HeightProfile, debug_frames: int = 0):
        self.profile = profile
        self._debug_frames_remaining = max(0, int(debug_frames))
        self._prev_tau: Optional[np.ndarray] = None
        self._filter_alpha = 0.3  # Low-pass filter coefficient (lower = more smoothing)
        self._contact_history: list[bool] = []
        self._q_des: Optional[np.ndarray] = None  # Desired joint pose captured from initial state

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
        phase, t_phase, phase_dur = phase_schedule(t)
        # Track contact stability (per-foot)
        left_contact, right_contact = foot_contacts(model, data)
        # Stance selection from phase (scripted gait); fall back to sensed contacts if both lost
        if phase == "left_stance":
            stance_left, stance_right = True, False
        elif phase == "right_stance":
            stance_left, stance_right = False, True
        else:  # double support
            stance_left, stance_right = True, True

        in_contact_any = stance_left or stance_right
        self._contact_history.append(in_contact_any)
        if len(self._contact_history) > 10:
            self._contact_history.pop(0)
        debug_this_frame = False
        if self._debug_frames_remaining > 0:
            debug_this_frame = True
            self._debug_frames_remaining -= 1
            print("\n[DEBUG][task2] ----- frame start -----")
            print(f"[DEBUG][task2] sim_time={t:.4f}s")
            print(f"[DEBUG][task2] phase={phase} t_phase={t_phase:.3f}/{phase_dur:.3f}")
            print(f"[DEBUG][task2] contact_left={left_contact} contact_right={right_contact}")
            print(f"[DEBUG][task2] stance_left={stance_left} stance_right={stance_right}")
            print(f"[DEBUG][task2] contact_history={self._contact_history[-5:]}")

        x = data.qpos[0]
        z = data.qpos[1]
        theta = data.qpos[2]
        xd = data.qvel[0]
        zd = data.qvel[1]
        thetad = data.qvel[2]

        q = data.qpos[3:7]
        qd = data.qvel[3:7]

        if debug_this_frame:
            print(f"[DEBUG][task2] root_pos=[x,z,theta]={_fmt_debug([x, z, theta])}")
            print(f"[DEBUG][task2] root_vel=[xd,zd,thetad]={_fmt_debug([xd, zd, thetad])}")
            print(f"[DEBUG][task2] joint_pos={_fmt_debug(q)}")
            print(f"[DEBUG][task2] joint_vel={_fmt_debug(qd)}")

        z_des, zd_des = self.profile.desired_height(t)
        x_des = 0.0
        theta_des = 0.0

        fx_des = Kp_root_xz * (x_des - x) + Kd_root_xz * (0.0 - xd)
        f_vert_des = Kp_root_xz * (z_des - z) + Kd_root_xz * (zd_des - zd)
        tau_theta = Kp_root_theta * (theta_des - theta) + Kd_root_theta * (0.0 - thetad)

        if debug_this_frame:
            print(f"[DEBUG][task2] desired_height={z_des:.4f} desired_height_vel={zd_des:.4f}")
            print(f"[DEBUG][task2] fx_des={fx_des:.4f} f_vert_des={f_vert_des:.4f} tau_theta={tau_theta:.4f}")

        # Capture desired joint pose from the initial state to avoid hardcoding
        if self._q_des is None:
            self._q_des = data.qpos[3:7].copy()
        q_des = self._q_des.copy()

        # Swing leg shaping (very simple joint-space offsets)
        def swing_offsets(progress: float) -> tuple[float, float]:
            # progress in [0,1]
            hip_off = STEP_LENGTH * (progress - 0.5) * 0.5  # small fore-aft hip swing (rad approx)
            knee_off = STEP_KNEE_BEND * math.sin(math.pi * progress)
            return hip_off, knee_off

        if phase == "left_stance":
            prog = max(0.0, min(1.0, t_phase / phase_dur))
            hip_off, knee_off = swing_offsets(prog)
            # Right leg is swing: joints 0,1
            q_des[0] += hip_off
            q_des[1] += knee_off
        elif phase == "right_stance":
            prog = max(0.0, min(1.0, t_phase / phase_dur))
            hip_off, knee_off = swing_offsets(prog)
            # Left leg is swing: joints 2,3
            q_des[2] += hip_off
            q_des[3] += knee_off

        tau_posture = Kp_joint * (q_des - q) - Kd_joint * qd

        tau_bias = data.qfrc_bias[3:7].copy()
        # Note: trunk Jacobian is zero for floating base (no actuation path from joints to trunk)
        # Base stabilization must come through contact forces in the QP, not tau_root
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_frame")
        mujoco.mj_jacBody(model, data, jacp, jacr, trunk_id)
        J_trunk = np.zeros((3, 4))
        J_trunk[0, :] = jacp[0, 3:7]
        J_trunk[1, :] = jacp[VERT_AXIS, 3:7]
        J_trunk[2, :] = jacr[1, 3:7]
        tau_root = J_trunk.T @ np.array([fx_des, f_vert_des, tau_theta])  # Will be ~zero
        tau_task = tau_root + tau_posture
        # Motor torques: tau_motor + tau_bias = tau_task, so tau_motor = tau_task - tau_bias
        tau_des = tau_task - tau_bias

        if debug_this_frame:
            print(f"[DEBUG][task2] q_des={_fmt_debug(q_des)}")
            print(f"[DEBUG][task2] tau_posture={_fmt_debug(tau_posture)}")
            print(f"[DEBUG][task2] tau_bias={_fmt_debug(tau_bias)}")
            print(f"[DEBUG][task2] trunk_jacobian=\n{_fmt_debug(J_trunk)}")
            print(f"[DEBUG][task2] tau_root={_fmt_debug(tau_root)}")
            print(f"[DEBUG][task2] tau_des={_fmt_debug(tau_des)}")

        jacp_r = np.zeros((3, model.nv))
        jacp_l = np.zeros((3, model.nv))
        jacr_tmp = np.zeros((3, model.nv))
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shin")
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shin")
        mujoco.mj_jacBody(model, data, jacp_r, jacr_tmp, right_id)
        mujoco.mj_jacBody(model, data, jacp_l, jacr_tmp, left_id)
        # Note: Jr vertical row may be near-zero at nominal pose (leg extended vertically)
        # causing QP to lean on left foot for vertical load. Consider wrench-balance formulation.
        Jr = np.vstack([jacp_r[0, 3:7], jacp_r[VERT_AXIS, 3:7]])
        Jl = np.vstack([jacp_l[0, 3:7], jacp_l[VERT_AXIS, 3:7]])
        if debug_this_frame:
            print(f"[DEBUG][task2] Jr=\n{_fmt_debug(Jr)}")
            print(f"[DEBUG][task2] Jl=\n{_fmt_debug(Jl)}")

        F = solve_contact_qp(
            Jr,
            Jl,
            tau_des,
            f_vert_des,
            contact_left=stance_left,
            contact_right=stance_right,
        )
        if debug_this_frame:
            print(f"[DEBUG][task2] contact_forces={_fmt_debug(F)}")
            print(f"[DEBUG][task2] f_vert_target={WEIGHT_FORCE + f_vert_des:.4f} (mg + f_vert_des)")
        Fx_r, Fy_r, Fx_l, Fy_l = F
        tau_contact = Jr.T @ np.array([Fx_r, Fy_r]) + Jl.T @ np.array([Fx_l, Fy_l])
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

        # Logging for plots
        self.log_time.append(t)
        self.log_root_pos.append(np.array([x, z, theta]))
        self.log_root_vel.append(np.array([xd, zd, thetad]))
        self.log_joint_pos.append(q.copy())
        self.log_joint_vel.append(qd.copy())
        self.log_tau_cmd.append(tau.copy())
        self.log_tau_raw.append(tau_raw.copy())
        self.log_tau_contact.append(tau_contact.copy())
        self.log_posture_boost.append(posture_boost.copy())
        self.log_contact_forces.append(np.array([Fx_r, Fy_r, Fx_l, Fy_l]))
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
        return tau, {"raw_tau": tau_raw}

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
