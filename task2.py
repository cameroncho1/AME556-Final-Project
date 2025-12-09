"""Task 2 standing controller using QP-based contact wrench allocation."""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from sim_runner import TORQUE_LIMITS, run_simulation

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
MU = 0.7  # Changed from 0.5 to match homework
FZ_MIN = 10.0  # Changed from 0.3 * WEIGHT_FORCE to match homework
FZ_MAX = 250.0  # Changed from 3.0 * WEIGHT_FORCE to match homework
MG_PENALTY_WEIGHT = 0.08  # Changed from 25.0 to match homework (was 315x too large!)
DEBUG_QP_FORCES = False
VERT_AXIS = 2  # MuJoCo's +Z is vertical axis

Kp_root_xz = 600.0
Kd_root_xz = 120.0
Kp_root_theta = 800.0
Kd_root_theta = 150.0
Kp_joint = 60.0  # Reduced from 120.0 to reduce oscillations
Kd_joint = 10.5  # Reduced from 25.0 to reduce oscillations


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
        if t < self.hold_time:
            return self.start_height, 0.0
        t2 = t - self.hold_time
        if t2 < self.rise_time:
            frac = t2 / self.rise_time
            height = self.start_height + frac * (self.peak_height - self.start_height)
            vel = (self.peak_height - self.start_height) / self.rise_time
            return height, vel
        t3 = t2 - self.rise_time
        if t3 < self.drop_time:
            frac = t3 / self.drop_time
            height = self.peak_height + frac * (self.final_height - self.peak_height)
            vel = (self.final_height - self.peak_height) / self.drop_time
            return height, vel
        return self.final_height, 0.0


def both_feet_in_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Check foot contacts using body IDs (geom names are absent in XML)."""
    floor_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground_body")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shin")
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shin")
    right = left = False
    # print(f"Left foot id {left_body}, right foot id {right_body}, floor id {floor_body}")
    for i in range(data.ncon):
        # print(f"Contact {i}: {data.contact[i]}")
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        if geom1 < 0 or geom2 < 0:
            continue
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]
        bodies = {body1, body2}
        # if right_body in bodies:
        #     print("Right foot contact detected.")
        #     print(f"Contact bodies: {bodies}")
        # if left_body in bodies:
        #     print("Left foot contact detected.")
        #     print(f"Contact bodies: {bodies}")
        if right_body in bodies and floor_body in bodies:
            right = True
        if left_body in bodies and floor_body in bodies:
            left = True
        if right and left:
            return True
    return False


def solve_contact_qp(Jr: np.ndarray, Jl: np.ndarray, tau_des: np.ndarray) -> np.ndarray:
    A = np.zeros((4, 4))
    A[:, 0:2] = Jr.T
    A[:, 2:4] = Jl.T
    Q = A.T @ A + 1e-6 * np.eye(4)
    p = -A.T @ tau_des

    v = np.array([0.0, 1.0, 0.0, 1.0])  # penalize deviations in vertical (Y) forces
    weight = MG_PENALTY_WEIGHT
    mg = WEIGHT_FORCE
    Q += weight * np.outer(v, v)
    p += -weight * mg * v

    if HAS_CVXOPT:
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
        try:
            sol = solvers.qp(P, q, G, h)
            F = np.array(sol["x"]).flatten()
            if DEBUG_QP_FORCES:
                Fx_r, Fy_r, Fx_l, Fy_l = F
                sum_fy = Fy_r + Fy_l
                print(
                    f"[QP] Fx_r={Fx_r:.2f} Fy_r={Fy_r:.2f} "
                    f"Fx_l={Fx_l:.2f} Fy_l={Fy_l:.2f} sumFy={sum_fy:.2f} mg={mg:.2f}"
                )
            return F
        except Exception:
            pass

    # Least-squares fallback
    print("[WARNING] QP solver failed, using least-squares fallback.")
    F_ls = np.linalg.pinv(A) @ tau_des
    Fx_r, Fy_r, Fx_l, Fy_l = F_ls
    Fy_r = np.clip(max(Fy_r, mg * 0.5), FZ_MIN, FZ_MAX)
    Fy_l = np.clip(max(Fy_l, mg * 0.5), FZ_MIN, FZ_MAX)
    Fx_r = np.clip(Fx_r, -MU * Fy_r, MU * Fy_r)
    Fx_l = np.clip(Fx_l, -MU * Fy_l, MU * Fy_l)
    return np.array([Fx_r, Fy_r, Fx_l, Fy_l])


class StandingQPController:
    def __init__(self, profile: HeightProfile, debug_frames: int = 0):
        self.profile = profile
        self._debug_frames_remaining = max(0, int(debug_frames))
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

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, t: float):
        # Track contact stability
        in_contact = both_feet_in_contact(model, data)
        self._contact_history.append(in_contact)
        if len(self._contact_history) > 10:
            self._contact_history.pop(0)

        debug_this_frame = False
        if self._debug_frames_remaining > 0:
            debug_this_frame = True
            self._debug_frames_remaining -= 1
            print("\n[DEBUG][task2] ----- frame start -----")
            print(f"[DEBUG][task2] sim_time={t:.4f}s")
            print(f"[DEBUG][task2] both_feet_in_contact={in_contact}")
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

        # Align q_des with keyframe init pose: [-0.5472, 0.5708, 0, 0.5708]
        # to avoid large initial posture torques that violate velocity limits
        q1_des = -0.5472  # right hip (was -π/3 ≈ -1.047)
        q2_des = 0.5708   # right knee (was π/2 ≈ 1.571)
        q3_des = 0.0      # left hip (was -π/6 ≈ -0.524)
        q4_des = 0.5708   # left knee (was π/2 ≈ 1.571)
        q_des = np.array([q1_des, q2_des, q3_des, q4_des])
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

        F = solve_contact_qp(Jr, Jl, tau_des)
        if debug_this_frame:
            print(f"[DEBUG][task2] contact_forces={_fmt_debug(F)}")
        Fx_r, Fy_r, Fx_l, Fy_l = F
        tau_contact = Jr.T @ np.array([Fx_r, Fy_r]) + Jl.T @ np.array([Fx_l, Fy_l])
        posture_boost = 0.15 * tau_posture  # Increased from 0.01 for better stability
        tau = tau_contact + posture_boost
        tau_raw = tau.copy()

        # Apply low-pass filter to smooth torque commands and reduce flickering
        if self._prev_tau is not None and in_contact:
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
        self.log_contact_flag.append(in_contact)

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

        return tau, {"raw_tau": tau_raw}

    def save_plots(self, output_dir: Optional[str] = None) -> None:
        if not self.log_time:
            print("[INFO] No logged data to plot.")
            return

        out_dir = output_dir or os.path.join(HERE, "plots")
        os.makedirs(out_dir, exist_ok=True)

        time_arr = np.asarray(self.log_time)
        root_pos = np.vstack(self.log_root_pos)
        root_vel = np.vstack(self.log_root_vel)
        joint_pos = np.vstack(self.log_joint_pos)
        joint_vel = np.vstack(self.log_joint_vel)
        tau_cmd = np.vstack(self.log_tau_cmd)
        tau_raw = np.vstack(self.log_tau_raw)
        tau_contact = np.vstack(self.log_tau_contact)
        posture_boost = np.vstack(self.log_posture_boost)
        contact_forces = np.vstack(self.log_contact_forces)
        height_des = np.vstack(self.log_height_des)
        contact_flag = np.asarray(self.log_contact_flag, dtype=float)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = StandingQPController(HeightProfile(), debug_frames=args.debug_frames)
    result = run_simulation(
        controller,
        sim_time=args.sim_time,
        interactive=args.interactive,
        perturb=args.perturb,
        description="Task 2 QP standing",
        stop_on_violation=not args.ignore_violations,
    )
    controller.save_plots(args.plots_dir)
    print(f"[INFO] Plots generated for simulation horizon t={result.final_time:.2f}s")


if __name__ == "__main__":
    main()
