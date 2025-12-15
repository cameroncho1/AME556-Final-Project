"""Quadratic Program (QP) solver for contact force allocation.
Extracted from task2.py for modular use.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco
import sim_runner as sim
# import sim_runner
from sim_runner import TrunkState, foot_contacts
import debug_tools as debug
try:
    from cvxopt import matrix, solvers  # type: ignore
    HAS_CVXOPT = True
    solvers.options["show_progress"] = False
except ImportError:
    HAS_CVXOPT = False

# QP solver constants
FZ_MIN = 0.0  # Minimum vertical force

# Weight-related constants for FZ_MAX calculation
MASS_TOTAL = 8 + 2 * 0.25  # 0.2 kg body + 4x0.25 kg legs
GRAVITY = 9.81
WEIGHT_FORCE = MASS_TOTAL * GRAVITY
FZ_MAX = 10 * WEIGHT_FORCE  # Maximum vertical force
POSTURE_RATIO = 6e-4  # Weighting for posture torque regularization

Kp_root_x = 0
Kd_root_x = 1000
Kp_root_z = 16000
Kd_root_z = 400
Kp_root_theta = 300
Kd_root_theta = 20
Kp_joint = 4000  # Reduced from 120.0 to reduce oscillations
Kd_joint = 200  # Reduced from 25.0 to reduce oscillations



def solve_contact_qp(
    G: np.ndarray,
    wrench_des: np.ndarray,
    contact_left: bool,
    contact_right: bool
) -> np.ndarray:
    """Solve QP (or LS fallback) for contact forces, supporting 1- or 2-foot contact.

    Returns forces ordered [Fx_r, Fz_r, Fx_l, Fz_l].
    G: grasp matrix (3 x n_forces), wrench_des: (3,)
    mu: friction coefficient
    fz_min: minimum vertical force
    fz_max: maximum vertical force
    """
    def _clip_force(Fx: float, Fz: float) -> tuple[float, float]:
        Fz_clipped = np.clip(Fz, -FZ_MIN, FZ_MAX)
        Fx_clipped = np.clip(Fx, -sim.MU * abs(Fz_clipped), sim.MU * abs(Fz_clipped))
        return Fx_clipped, Fz_clipped
    
    if not contact_left and not contact_right:
        print("[WARN][qp_solver] No contacts detected; returning zero forces.")
        return np.zeros(4)

    # Left-only contact
    if contact_left and not contact_right:
        G_left = G[:, 2:4]
        Q = G_left.T @ G_left + 1e-20 * np.eye(2)
        p = -G_left.T @ wrench_des
        if HAS_CVXOPT:
            try:
                P = matrix(2.0 * Q)
                q = matrix(2.0 * p)
                Gc = matrix(
                    np.array(
                        [
                            [1.0, -sim.MU],
                            [-1.0, -sim.MU],
                            [0.0, -1.0],
                            [0.0, 1.0],
                        ],
                        dtype=float,
                    )
                )
                h = matrix(np.array([0.0, 0.0, -FZ_MIN, FZ_MAX], dtype=float))
                sol = solvers.qp(P, q, Gc, h)
                fx_l, fz_l = np.array(sol["x"]).flatten()
                print(f"[DEBUG][qp_solver] CVXOPT left contact solution: fx_l={fx_l:.2f}, fz_l={fz_l:.2f}")
            except Exception:
                fx_l, fz_l = np.linalg.lstsq(G_left, wrench_des, rcond=None)[0]
        else:
            fx_l, fz_l = np.linalg.lstsq(G_left, wrench_des, rcond=None)[0]
        fx_l, fz_l = _clip_force(fx_l, fz_l)
        F = -np.array([0.0, 0.0, fx_l, fz_l])
        return F

    # Right-only contact
    if contact_right and not contact_left:
        G_right = G[:, 0:2]
        Q = G_right.T @ G_right + 1e-20 * np.eye(2)
        p = -G_right.T @ wrench_des
        if HAS_CVXOPT:
            try:
                P = matrix(2.0 * Q)
                q = matrix(2.0 * p)
                Gc = matrix(
                    np.array(
                        [
                            [1.0, -sim.MU],
                            [-1.0, -sim.MU],
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
        F = -np.array([fx_r, fz_r, 0.0, 0.0])
        return F

    # Both feet in contact (default 4-force problem)
    Q = G.T @ G + 1e-20 * np.eye(4)
    p = -G.T @ wrench_des

    if HAS_CVXOPT:
        try:
            P = matrix(2.0 * Q)
            q = matrix(2.0 * p)
            Gc = matrix(
                np.array(
                    [
                        [1.0, -sim.MU, 0.0, 0.0],
                        [-1.0, -sim.MU, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -sim.MU],
                        [0.0, 0.0, -1.0, -sim.MU],
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
            F = -np.array(sol["x"]).flatten()
            return F
        except Exception:
            pass

    # Least-squares fallback (both feet)
    F_ls = np.linalg.lstsq(G, wrench_des, rcond=None)[0]
    if F_ls.shape[0] < 4:
        F_ls = np.pad(F_ls, (0, 4 - F_ls.shape[0]))
    Fx_r, Fz_r, Fx_l, Fz_l = F_ls[:4]
    Fx_r, Fz_r = _clip_force(Fx_r, Fz_r)
    Fx_l, Fz_l = _clip_force(Fx_l, Fz_l)
    F = -np.array([Fx_r, Fz_r, Fx_l, Fz_l])
    return F

VERT_AXIS = 2  # MuJoCo's +Z is vertical axis


@dataclass
class QPControllerResult:
    """Result from QP controller computation."""
    tau: np.ndarray  # (4,) total joint torques
    tau_contact: np.ndarray  # (4,) contact torques
    contact_forces: np.ndarray  # (4,) [Fx_r, Fz_r, Fx_l, Fz_l]
    wrench_des: np.ndarray  # (3,) desired wrench [fx, fz, tau_theta]
    G: np.ndarray  # (3, 4) grasp matrix


def qp_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos_des: np.ndarray,  # (3,) desired position [x, y, z]
    vel_des: np.ndarray,  # (3,) desired velocity [xd, yd, zd]
    theta_des: float,  # desired pitch angle
    q_des: np.ndarray,
    qd_des: np.ndarray,
    left_enable: bool = True,
    right_enable: bool = True,
) -> QPControllerResult:
    """QP-based contact force controller.
    
    Computes PD control for root, builds grasp matrix, solves QP, and returns contact torques.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        trunk_state: Current trunk state
        pos_des: Desired position [x, y, z]
        vel_des: Desired velocity [xd, yd, zd]
        theta_des: Desired pitch angle
        contact_left: Left foot in contact
        contact_right: Right foot in contact
        kp_root_x, kp_root_z, kd_root_x, kd_root_z: PD gains for position
        kp_root_theta, kd_root_theta: PD gains for orientation
        
    Returns:
        QPControllerResult with tau_contact, contact_forces, wrench_des, and G matrix
    """
    
    # Track contact stability (per-foot)
    contact_left, contact_right = foot_contacts(model, data)

    # Get trunk state using helper function
    trunk_state = sim.get_trunk_state(model, data, "body_frame")
    trunk_pos = trunk_state.pos
    trunk_vel = trunk_state.vel
    trunk_angvel = trunk_state.angvel
    pitch = trunk_state.theta
    pitch_rate = trunk_state.thetad
    
    # Debug visualization
    # print("[DEBUG][task2] trunk_pos calculation:", debug._fmt_debug(trunk_pos))
    debug.draw_world_position(sim.get_viewer(), trunk_pos, np.array([1.0, 1.0, 1.0, 0.1]))
    # print("[DEBUG][task2] trunk_vel calculation:", debug._fmt_debug(trunk_vel))
    debug.draw_arrow(
        sim.get_viewer(),
        trunk_pos,
        trunk_pos + trunk_vel * 1,
        np.array([1.0, 1.0, 1.0, 0.6]),
    )
    # print("[DEBUG][task2] trunk_angvel:", debug._fmt_debug(trunk_angvel))
    # print("[DEBUG][task2] pitch calculation:", pitch)
    # print("[DEBUG][task2] pitch_rate calculation:", pitch_rate)

    # PD control for root position
    fz_des = Kp_root_z * (pos_des[2] - trunk_pos[2]) + Kd_root_z * (vel_des[2] - trunk_vel[2])
    fx_des = Kp_root_x * (pos_des[0] - trunk_pos[0]) + Kd_root_x * (vel_des[0] - trunk_vel[0])
    # print("[DEBUG][qp_solver] fx_des calculation: {:.4f}  fz_des calculation: {:.4f}".format(fx_des, fz_des))
    # PD control for pitch
    tau_theta = -(Kp_root_theta * (theta_des - pitch) + Kd_root_theta * (0.0 - pitch_rate))
    debug.draw_arrow(
        sim.get_viewer(),
        trunk_pos,
        trunk_pos + np.array([fx_des, 0.0, fz_des]) * 0.01,
        np.array([0, 1.0, 0.5, 0.6]),
    )
    debug.draw_arrow(
        sim.get_viewer(),
        trunk_pos,
        trunk_pos + np.array([0.0, tau_theta * 0.01, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.6]),
    )

    # Desired trunk wrench: [fx_des, fz_des, tau_theta]
    wrench_des = np.array([fx_des, fz_des, tau_theta])
    
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
    G = np.zeros((3, 4))
    G[0, 0] = 1.0  # Fx_r
    G[1, 1] = 1.0  # Fz_r
    G[0, 2] = 1.0  # Fx_l
    G[1, 3] = 1.0  # Fz_l
    
    # Pitch (about Y): moment arm from foot to trunk
    right_foot_pos = data.xpos[right_id].copy()
    left_foot_pos = data.xpos[left_id].copy()
    trunk_pos = trunk_state.pos.copy()
    
    # Right foot
    dx_r = right_foot_pos[0] - trunk_pos[0]
    dz_r = right_foot_pos[2] - trunk_pos[2]
    # Left foot
    dx_l = left_foot_pos[0] - trunk_pos[0]
    dz_l = left_foot_pos[2] - trunk_pos[2]
    
    G[2, 0] = -dz_r  # Fx_r moment arm
    G[2, 1] = dx_r   # Fz_r moment arm
    G[2, 2] = -dz_l  # Fx_l moment arm
    G[2, 3] = dx_l   # Fz_l moment arm
    
    # Solve QP for contact forces
    F = solve_contact_qp(
        G, 
        wrench_des, 
        contact_left and left_enable,
        contact_right and right_enable
        )
    # F = -F  # Negate to convert to MuJoCo frame convention
    Fx_r, Fz_r, Fx_l, Fz_l = F
    info = {"contact_forces": np.array([Fx_r, Fz_r, Fx_l, Fz_l]), 
            "right_contact": contact_right, 
            "left_contact": contact_left
            }
    print(f"[DEBUG][qp_solver] Contact forces: Fx_r={Fx_r:.2f}, Fz_r={Fz_r:.2f}, Fx_l={Fx_l:.2f}, Fz_l={Fz_l:.2f}")
    debug.draw_contact_force_arrows(
        sim.get_viewer(),
        model,
        data,
        info
    )
    # Compute contact torques from forces
    tau_contact = Jr.T @ np.array([Fx_r, Fz_r]) + Jl.T @ np.array([Fx_l, Fz_l])
    
    # q_des = [-1.2471975512, 1.0707963268, -0.2, 1.0707963268]  # desired joint positions
    # qd_des = np.zeros(4)  # desired joint velocities
    tau_posture = np.zeros(4)
    for i in range(4):
        q_curr = data.qpos[3 + i]
        qd_curr = data.qvel[3 + i]
        tau_posture[i] = (
            Kp_joint * (q_des[i] - q_curr) + Kd_joint * (qd_des[i] - qd_curr)
        )
    # Posture boost to help maintain upright pose
    tau_bias = data.qfrc_bias[3:7].copy()
    
    # if debug_this_frame:
    #     print(f"[DEBUG][task2] tau_bias={_fmt_debug(tau_bias)}")
    posture_boost = POSTURE_RATIO * tau_posture  # Increased from 0.01 for better stability
    tau = tau_contact + posture_boost
    tau_raw = tau.copy()

    # # Apply low-pass filter to smooth torque commands and reduce flickering
    # if self._prev_tau is not None and in_contact_any:
    #     # tau_filtered = self._filter_alpha * tau_raw + (1 - self._filter_alpha) * self._prev_tau
    #     tau_filtered = tau_raw.copy()
    # else:
    #     tau_filtered = tau_raw

    # self._prev_tau = tau_raw.copy()
    # Clip to per-joint torque limits to avoid constraint violations
    tau = np.clip(tau_raw, -sim.TORQUE_LIMITS, sim.TORQUE_LIMITS)

    # Debug draw tau
    if(False):
        joint_ids = [3, 4, 6, 7]
        for i in range(4):
            joint_pos = data.xpos[joint_ids[i]]
            debug.draw_arrow(
                sim.get_viewer(),
                joint_pos,
                joint_pos + np.array([0.0, tau[i] * 0.1, 0.0]),
                np.array([0.0, 0.4, 1.0, 0.6]),
            )

    return QPControllerResult(
        tau,
        tau_contact=tau_contact,
        contact_forces=np.array([Fx_r, Fz_r, Fx_l, Fz_l]),
        wrench_des=wrench_des,
        G=G
    )
