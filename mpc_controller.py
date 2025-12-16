"""
Model-predictive trunk controller that predicts CoM dynamics (LIPM)
and allocates the desired wrench through the existing QP contact solver.

The controller:
- predicts CoM evolution over a short horizon using a linear inverted
  pendulum model (LIPM) and planned foot placements,
- computes a desired wrench (fx, fz, tau_pitch) that steers the first
  step of the predicted trajectory, and
- feeds that wrench into the QP contact allocator to obtain joint
  torques.

This file intentionally lives alongside the simpler `mpc_walking_controller.py`
but does not modify that code. Use this controller when you want true
MPC-style prediction of the trunk dynamics instead of a single-step
heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import mujoco
import numpy as np

import sim_runner as sim
from sim_runner import MU, TORQUE_LIMITS, TrunkState, foot_contacts, get_trunk_state
from qp_solver import (
    QPControllerResult,
    POSTURE_RATIO,
    MASS_TOTAL,
    GRAVITY,
    solve_contact_qp,
)
import debug_tools as debug


# Gains for MPC tracking of trunk state
if False:
    Kp_x = 0.0
    Kd_x = 600.0
    Kp_z = 1000.0
    Kd_z = 10.0
    Kp_theta = 3000.0
    Kd_theta = 10.0
    Kp_joint = 400.0
    Kd_joint = 10.0
else:

    Kp_root_x = 0
    Kd_root_x = 50
    Kp_root_z = 400
    Kd_root_z = 10
    Kp_root_theta = 1000
    Kd_root_theta = 20
    Kp_joint = 2000
    Kd_joint = 30

# Desired joint posture (same convention as qp_solver)
Q_DES_DEFAULT = np.array([-1.2471975512, 1.0707963268, -0.2, 1.0707963268])
QD_DES_DEFAULT = np.zeros(4)


@dataclass
class LIPMParams:
    """Parameters for the linear inverted pendulum dynamics."""

    com_height: float = 0.5
    dt: float = 0.05

    @property
    def omega(self) -> float:
        return float(np.sqrt(GRAVITY / self.com_height))


@dataclass
class FootstepPlan:
    """Holds planned footsteps over the horizon."""

    swing_sequence: List[str] = field(default_factory=list)
    footsteps: List[np.ndarray] = field(default_factory=list)

    def first_target(self) -> np.ndarray:
        return self.footsteps[0] if self.footsteps else np.zeros(3)


class FootstepMPCPlanner:
    """
    Simple horizon footstep planner that uses capture-point style targets
    but keeps a short horizon of alternating foot placements.
    """

    def __init__(
        self,
        step_time: float = 0.4,
        horizon_steps: int = 3,
        step_width: float = 0.18,
        v_des: float = 0.5,
    ) -> None:
        self.step_time = step_time
        self.horizon_steps = horizon_steps
        self.step_width = step_width
        self.v_des = v_des
        self.reset()

    def reset(self) -> None:
        self.swing = "left"
        self.next_switch = self.step_time
        self.last_switch = 0.0
        self.cached_plan: FootstepPlan = FootstepPlan()

    def update_phase(self, t: float) -> None:
        if t >= self.next_switch:
            self.swing = "right" if self.swing == "left" else "left"
            self.last_switch = self.next_switch
            self.next_switch += self.step_time

    def phase(self, t: float) -> float:
        """Return normalized swing phase in [0,1]."""
        return float(np.clip((t - self.last_switch) / self.step_time, 0.0, 1.0))

    def current_swing(self) -> str:
        return self.swing

    def build_plan(
        self,
        trunk_state: TrunkState,
        omega: float,
        left_pos: np.ndarray,
        right_pos: np.ndarray,
    ) -> FootstepPlan:
        """
        Build an alternating sequence of footsteps starting from the current swing foot.
        Uses a capture-point inspired x target and fixed lateral offsets.
        """
        # Dynamics-aware foot placement (capture-point style) with velocity-error feedback.
        #
        # - capture point: reacts to current (x, xd) so that v_des=0 while moving lands to stop.
        # - + v_des * step_time: feedforward bias to keep walking at commanded speed.
        # - + k_vel * (v_des - xd) * step_time: feedback so overspeed/underspeed changes step length.
        #
        # Intuition:
        # - if xd < v_des: step a bit farther forward to speed up
        # - if xd > v_des: step not as far forward (relative to the feedforward) to slow down
        k_vel = 0.2
        omega_safe = max(float(omega), 1e-6)
        capture_point = trunk_state.x + trunk_state.xd / omega_safe
        base_target_x = -0.1 * self.step_time + capture_point - (0.15* self.v_des  + k_vel * (self.v_des - trunk_state.xd)) * self.step_time
        print(f"[DEBUG][MPC] v_des={self.v_des}, xd={trunk_state.xd}, target_x={base_target_x}")
        swing_seq: List[str] = []
        footsteps: List[np.ndarray] = []
        swing = self.swing

        for k in range(self.horizon_steps):
            y_off = self.step_width * 0.5 if swing == "left" else -self.step_width * 0.5
            step_x = base_target_x + k * self.v_des * self.step_time
            footsteps.append(np.array([step_x, y_off, 0.0]))
            swing_seq.append(swing)
            swing = "right" if swing == "left" else "left"

        self.cached_plan = FootstepPlan(swing_seq, footsteps)
        return self.cached_plan


class ModelPredictiveController:
    """
    High-level MPC controller that predicts trunk dynamics, computes a desired wrench,
    and allocates contact forces through the existing QP contact allocator.
    """

    def __init__(
        self,
        lipm: LIPMParams | None = None,
        planner: FootstepMPCPlanner | None = None,
        swing_height: float = 0.12,
        theta_des: float = 0.0,
        q_des: np.ndarray | None = None,
        qd_des: np.ndarray | None = None,
    ) -> None:
        self.lipm = lipm or LIPMParams()
        self.planner = planner or FootstepMPCPlanner()
        self.swing_height = swing_height
        self.theta_des = theta_des
        self.q_des = q_des if q_des is not None else Q_DES_DEFAULT.copy()
        self.qd_des = qd_des if qd_des is not None else QD_DES_DEFAULT.copy()
        self._prev_swing: str | None = None
        # Swing start expressed in the *current trunk/body frame* at liftoff.
        self._swing_start_body: np.ndarray | None = None
        self._swing_target: np.ndarray | None = None
        self._swing_com_start_x: float | None = None
        # (Debug drawing only) We intentionally do NOT use MuJoCo FK for drawing q_des_use;
        # we compute the leg points manually from q and trunk pose.

    @staticmethod
    def _roty(a: float) -> np.ndarray:
        ca = float(np.cos(a))
        sa = float(np.sin(a))
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=float)

    def _world_to_body(self, trunk_state: TrunkState, p_world: np.ndarray) -> np.ndarray:
        """Convert a world point to trunk/body-frame coordinates (pitch-only model)."""
        Rb = self._roty(trunk_state.theta)
        return Rb.T @ (p_world - trunk_state.pos)

    def _body_to_world(self, trunk_state: TrunkState, p_body: np.ndarray) -> np.ndarray:
        """Convert a trunk/body-frame point to world coordinates (pitch-only model)."""
        Rb = self._roty(trunk_state.theta)
        return trunk_state.pos + (Rb @ p_body)

    def _predict_traj(
        self,
        trunk_state: TrunkState,
        foot_seq: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rollout LIPM dynamics over the horizon using the provided foot x positions.
        Returns arrays of predicted x and xd for each step (length = len(foot_seq)).
        """
        x, xd = trunk_state.x, trunk_state.xd
        omega = self.lipm.omega
        dt = self.lipm.dt
        x_pred = []
        xd_pred = []
        for p in foot_seq:
            x = x + xd * dt
            xd = xd + omega * omega * (x - p) * dt
            x_pred.append(x)
            xd_pred.append(xd)
        return np.array(x_pred), np.array(xd_pred)

    def _build_foot_sequence(
        self,
        plan: FootstepPlan,
        left_pos: np.ndarray,
        right_pos: np.ndarray,
    ) -> List[float]:
        """
        Build a stance-foot x sequence for the LIPM rollout.
        The first element is the current stance x, followed by planned swing landings.
        """
        # Current stance foot assumed to be the one with contact; fallback to average
        left_x = float(left_pos[0])
        right_x = float(right_pos[0])
        current_stance_x = 0.5 * (left_x + right_x)

        foot_seq: List[float] = [current_stance_x]
        for step in plan.footsteps:
            foot_seq.append(float(step[0]))
        # Ensure sequence length matches horizon
        return foot_seq[: self.planner.horizon_steps]

    def _compute_wrench(
        self,
        trunk_state: TrunkState,
        x_ref: float,
        xd_ref: float,
        stance_x: float,
    ) -> np.ndarray:
        """Compute desired wrench (fx, fz, tau_theta) for the trunk."""
        omega = self.lipm.omega
        # Feedforward to track LIPM dynamics plus PD for robustness
        fx_ff = MASS_TOTAL * omega * omega * (trunk_state.x - stance_x)
        fx_pd = Kp_root_x * (x_ref - trunk_state.x) + Kd_root_x * (xd_ref - trunk_state.xd)
        fx_des = fx_ff + fx_pd
        # fx_des = Kp_root_x * (x_ref - trunk_state.x) + Kd_root_x * (xd_ref - trunk_state.xd)
        fz_des = MASS_TOTAL * GRAVITY + Kp_root_z * (self.lipm.com_height - trunk_state.z) + Kd_root_z * (0.0 - trunk_state.zd)
        tau_theta = (Kp_root_theta * (self.theta_des - trunk_state.theta) + Kd_root_theta * (0.0 - trunk_state.thetad))
        return np.array([fx_des, fz_des, tau_theta])

    def _allocate_contact(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        wrench_des: np.ndarray,
        contact_left: bool,
        contact_right: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Allocate the desired wrench into contact forces using the existing QP solver
        and map them to joint torques via Jacobians.
        """
        jacp_r = np.zeros((3, model.nv))
        jacp_l = np.zeros((3, model.nv))
        jacr_tmp = np.zeros((3, model.nv))
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        mujoco.mj_jacBody(model, data, jacp_r, jacr_tmp, right_id)
        mujoco.mj_jacBody(model, data, jacp_l, jacr_tmp, left_id)
        Jr = np.vstack([jacp_r[0, 3:7], jacp_r[2, 3:7]])
        Jl = np.vstack([jacp_l[0, 3:7], jacp_l[2, 3:7]])

        # Build grasp matrix
        right_pos = data.xpos[right_id].copy()
        left_pos = data.xpos[left_id].copy()
        trunk_pos = get_trunk_state(model, data, "body_frame").pos.copy()

        dx_r = right_pos[0] - trunk_pos[0]
        dz_r = right_pos[2] - trunk_pos[2]
        dx_l = left_pos[0] - trunk_pos[0]
        dz_l = left_pos[2] - trunk_pos[2]

        G = np.zeros((3, 4))
        G[0, 0] = 1.0
        G[1, 1] = 1.0
        G[0, 2] = 1.0
        G[1, 3] = 1.0
        G[2, 0] = dz_r
        G[2, 1] = -dx_r
        G[2, 2] = dz_l
        G[2, 3] = -dx_l

        contact_forces = solve_contact_qp(G, wrench_des, contact_left, contact_right)
        Fx_r, Fz_r, Fx_l, Fz_l = contact_forces
        tau_contact = Jr.T @ np.array([Fx_r, Fz_r]) + Jl.T @ np.array([Fx_l, Fz_l])
        return tau_contact, contact_forces, G, Jr, Jl

    def _posture_torque(self, data: mujoco.MjData, q_des: np.ndarray, qd_des: np.ndarray) -> np.ndarray:
        """Joint-space PD to keep legs near a nominal posture (or an MPC-provided joint target)."""
        tau_posture = np.zeros(4)
        for i in range(4):
            q_curr = data.qpos[3 + i]
            qd_curr = data.qvel[3 + i]
            tau_posture[i] = (q_des[i] - q_curr) * Kp_joint + (qd_des[i] - qd_curr) * Kd_joint
        return tau_posture

    def _draw_leg_q_des(
        self,
        viewer: mujoco.viewer.Handle,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_des_use: np.ndarray,
        *,
        leg: str,
        color: np.ndarray,
    ) -> None:
        """
        Draw the desired leg configuration implied by `q_des_use` by manually computing
        the planar (x,z) forward kinematics from joint angles and trunk pose.

        Kinematics are derived from `biped_robot.xml`:
        - hip offset from trunk: (0, ±0.04, -0.125)
        - thigh length: 0.22, shin length: 0.22
        - foot offset from shin end: (0, ±0.01, -0.22)
        All rotations are about +Y (pitch), so motion is in the X–Z plane.
        """
        trunk = get_trunk_state(model, data, "body_frame")
        Rb = self._roty(trunk.theta)
        trunk_pos = trunk.pos

        # Link geometry (meters), from XML
        L1 = 0.22
        L2 = 0.22
        hip_z = -0.125
        hip_y = 0.04 if leg == "left" else -0.04
        foot_y = 0.01 if leg == "left" else -0.01

        # Joint angles from q_des_use (qpos[3:7] ordering: [q1,q2,q3,q4])
        if leg == "left":
            qh = float(q_des_use[0])
            qk = float(q_des_use[1])
        else:
            qh = float(q_des_use[2])
            qk = float(q_des_use[3])

        Rh = self._roty(qh)
        Rk = self._roty(qk)

        # Hip (thigh frame origin) in world
        p_hip = trunk_pos + Rb @ np.array([0.0, hip_y, hip_z], dtype=float)
        # Knee in world
        p_knee = p_hip + (Rb @ (Rh @ np.array([0.0, 0.0, -L1], dtype=float)))
        # Foot in world (end of shin + small lateral offset)
        p_foot = p_knee + (Rb @ (Rh @ (Rk @ np.array([0.0, foot_y, -L2], dtype=float))))

        debug.draw_arrow(viewer, p_hip, p_knee, color)
        debug.draw_arrow(viewer, p_knee, p_foot, color)
        debug.draw_world_position(viewer, p_hip, color, size=0.02)
        debug.draw_world_position(viewer, p_knee, color, size=0.02)
        debug.draw_world_position(viewer, p_foot, color, size=0.02)

    def _ik_swing_leg_body(
        self,
        data: mujoco.MjData,
        swing: str,
        foot_body_des: np.ndarray,
        *,
        iters: int = 12,
        damping: float = 10,
        step: float = 0.6,
    ) -> tuple[float, float]:
        """Simple closed-form 2-link IK in the sagittal (x,z) plane.

        We solve for (hip, knee) such that the foot reaches `foot_body_des` in the
        trunk/body frame, using the known link lengths from `biped_robot.xml`:
        - thigh length L1 = 0.22
        - shin  length L2 = 0.22
        Hip offset from trunk: (0, ±0.04, -0.125).

        Note: `iters/damping/step` are kept for API compatibility but are unused.
        """
        # Geometry from XML
        L1 = 0.22
        L2 = 0.22
        hip_off = np.array([0.0, 0.04 if swing == "left" else -0.04, -0.125], dtype=float)

        # Relative to hip joint (in body frame)
        p_rel = foot_body_des - hip_off
        x = float(p_rel[0])
        z = float(p_rel[2])

        # Convert to a plane where "down" is positive to match link default direction (-Z)
        # u = L*sin(q), v = L*cos(q) with u=-x, v=-z
        u = -x
        v = -z

        r2 = u * u + v * v
        r = float(np.sqrt(max(r2, 1e-12)))
        # Clamp to reachable annulus
        r = float(np.clip(r, abs(L1 - L2) + 1e-6, (L1 + L2) - 1e-6))
        r2 = r * r

        cos_k = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        cos_k = float(np.clip(cos_k, -1.0, 1.0))
        k_abs = float(np.arccos(cos_k))  # knee bend magnitude

        phi = float(np.arctan2(u, v))
        def _solve(qk: float) -> tuple[float, float]:
            psi = float(np.arctan2(L2 * np.sin(qk), L1 + L2 * np.cos(qk)))
            qh = phi - psi
            return qh, qk

        # Choose elbow configuration closest to current joint angles for continuity
        q_curr = data.qpos[3:7].copy()
        if swing == "left":
            qh0, qk0 = float(q_curr[0]), float(q_curr[1])
        else:
            qh0, qk0 = float(q_curr[2]), float(q_curr[3])

        sol1 = _solve(+k_abs)
        sol2 = _solve(-k_abs)
        def _cost(sol: tuple[float, float]) -> float:
            return abs(sol[0] - qh0) + abs(sol[1] - qk0)
        qh, qk = sol1 if _cost(sol1) <= _cost(sol2) else sol2
        return float(qh), float(qk)

    def _swing_torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        swing: str,
        swing_target: np.ndarray,
    ) -> np.ndarray:
        """Swing-foot PD mapped to joint torques.

        This version is *lift-only*: it only regulates Z (vertical) to keep the swing
        foot off the ground, and does not command forward/backward (X) motion.
        """
        Kp_sw = 400.0
        Kd_sw = 4.0
        jacp = np.zeros((3, model.nv))
        jacr_tmp = np.zeros((3, model.nv))
        body_name = f"{swing}_foot"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_jacBody(model, data, jacp, jacr_tmp, body_id)
        J = np.vstack([jacp[0, 3:7], jacp[2, 3:7]])  # x and z rows
        foot_pos = data.xpos[body_id].copy()
        foot_vel = data.cvel[body_id][3:].copy()
        err_z = float(swing_target[2] - foot_pos[2])
        f_sw = np.array([0.0, Kp_sw * err_z - Kd_sw * foot_vel[2]], dtype=float)
        tau_sw = J.T @ f_sw
        # qpos[3:7] ordering: [q1,q2,q3,q4] = [hip_L,knee_L,hip_R,knee_R]
        if swing == "left":
            return np.array([-tau_sw[0], -tau_sw[1], 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, -tau_sw[2], -tau_sw[3]])

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, t: float) -> QPControllerResult:
        # Phase update and contact state
        self.planner.update_phase(t)
        contact_left, contact_right = foot_contacts(model, data)
        trunk_state = get_trunk_state(model, data, "body_frame")

        # Current foot positions
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        right_pos = data.xpos[right_id].copy()
        left_pos = data.xpos[left_id].copy()

        # Build/refresh plan
        plan = self.planner.build_plan(trunk_state, self.lipm.omega, left_pos, right_pos)
        foot_seq = self._build_foot_sequence(plan, left_pos, right_pos)
        x_pred, xd_pred = self._predict_traj(trunk_state, foot_seq)

        # Reference is the first predicted step
        x_ref = x_pred[0] if len(x_pred) > 0 else trunk_state.x
        xd_ref = xd_pred[0] if len(xd_pred) > 0 else trunk_state.xd
        stance_x = foot_seq[0] if len(foot_seq) > 0 else trunk_state.x

        x_ref = trunk_state.x
        # xd_ref = 0.0
        # xd_ref = 0.
        # xd
        wrench_des = self._compute_wrench(trunk_state, x_ref, xd_ref, stance_x)
        # wrench_des
        # Swing foot tracking
        swing = self.planner.current_swing()

        tau_contact, contact_forces, G, Jr, Jl = self._allocate_contact(
            model, data, wrench_des, 
            # contact_left and swing != "left",
            # contact_right and swing != "right", 
            contact_left,
            contact_right
        )
        if swing != self._prev_swing:
            # New swing phase; latch start + target based on *current* dynamics.
            # This must NOT run every timestep, otherwise the swing target will "jump".
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{swing}_foot")
            foot_world = data.xpos[body_id].copy()
            self._swing_start_body = self._world_to_body(trunk_state, foot_world)
            # Latch the swing landing target at liftoff (prevents mid-swing target drift)
            self._swing_target = plan.footsteps[0].copy() if plan.footsteps else np.zeros(3)
            self._swing_target[2] = 0.0
            # Latch CoM x at liftoff for a dynamics-based phase variable
            self._swing_com_start_x = trunk_state.x
            self._prev_swing = swing
        phase_time = self.planner.phase(t)
        if self._swing_start_body is None:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{swing}_foot")
            foot_world = data.xpos[body_id].copy()
            self._swing_start_body = self._world_to_body(trunk_state, foot_world)
        if self._swing_target is None:
            self._swing_target = plan.footsteps[0].copy() if plan.footsteps else np.zeros(3)
            self._swing_target[2] = 0.0
        if self._swing_com_start_x is None:
            self._swing_com_start_x = trunk_state.x

        # MPC decides desired foot placement (use the planned footstep target).
        # We generate a *dynamics-based* swing phase variable using CoM progress.
        # The swing trajectory itself is represented in the *trunk/body frame* (local).
        swing_target_world = self._swing_target.copy()
        swing_target_body = self._world_to_body(trunk_state, swing_target_world)
        start_body = self._swing_start_body

        denom = float(swing_target_world[0] - self._swing_com_start_x)
        if abs(denom) < 1e-6:
            phase_dyn = phase_time
        else:
            phase_dyn = float(np.clip((trunk_state.x - self._swing_com_start_x) / denom, 0.0, 1.0))
        # Ensure the swing does not stall if the body isn't progressing as expected.
        # This keeps "dynamic-based" behavior but enforces a minimum rate to finish the step.
        phase_dyn = max(phase_dyn, phase_time)
        swing_des_body = start_body + (swing_target_body - start_body) * phase_dyn
        swing_des_body[2] = swing_target_body[2] + self.swing_height * 4.0 * phase_dyn * (1.0 - phase_dyn)
        swing_des = self._body_to_world(trunk_state, swing_des_body)

        # MPC-driven joint targets: solve IK for swing leg so the posture term supports the swing motion.
        q_des_use = self.q_des.copy()
        qd_des_use = self.qd_des.copy()
        hip_tgt, knee_tgt = self._ik_swing_leg_body(data, swing, swing_des_body)
        # qpos[3:7] ordering: [q1,q2,q3,q4] = [hip_L,knee_L,hip_R,knee_R]
        if swing == "left":
            q_des_use[0] = hip_tgt
            q_des_use[1] = knee_tgt
            q_des_use[2] = -0.2
            q_des_use[3] = 1.0707963268
        else:
            q_des_use[0] = -1.2471975512
            q_des_use[1] = 1.0707963268
            q_des_use[2] = hip_tgt
            q_des_use[3] = knee_tgt
        
        # q_des_use = [-1.2471975512, 1.0707963268, -0.2, 1.0707963268]
        dt = float(model.opt.timestep)
        q_curr = data.qpos[3:7].copy()
        # qd_des_use = (q_des_use - q_curr) #/ max(dt, 1e-6) 
        qd_des_use = np.zeros(4)
        tau_posture = self._posture_torque(data, q_des_use, qd_des_use) * POSTURE_RATIO
        # Boost swing-leg tracking so it can land fast enough to "catch" the COM when stopping.
        SWING_POSTURE_BOOST = 6.0
        if swing == "left":
            tau_posture[0:2] *= SWING_POSTURE_BOOST
        else:
            tau_posture[2:4] *= SWING_POSTURE_BOOST
            
        tau = tau_contact + tau_posture
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

        # Debug visualization
        viewer = sim.get_viewer()
        if viewer:
            debug.draw_tau(
                viewer,
                data,
                tau
            )
            # Draw desired joint configuration for the swing leg (from q_des_use)
            self._draw_leg_q_des(
                viewer,
                model,
                data,
                q_des_use,
                leg=swing,
                color=np.array([0.8, 0.2, 1.0, 0.6]),
            )
            # Draw desired wrench on trunk
            debug.draw_arrow(
                viewer,
                trunk_state.pos,
                trunk_state.pos + np.array([wrench_des[0], 0.0, wrench_des[1]]) * 0.01,
                np.array([0.0, 0.8, 0.2, 0.6]),
            )
            # Draw planned footstep targets (horizon)
            for f in plan.footsteps:
                debug.draw_world_position(viewer, f, np.array([0.2, 0.6, 1.0, 0.5]))
            # Draw latched landing target (first step) in red for clarity
            if self._swing_target is not None:
                debug.draw_world_position(viewer, self._swing_target, np.array([1.0, 0.0, 0.0, 0.6]), size=0.035)
            # Draw predicted CoM samples along horizon
            for xp in x_pred:
                com_pt = np.array([xp, trunk_state.pos[1], self.lipm.com_height])
                debug.draw_world_position(viewer, com_pt, np.array([1.0, 0.8, 0.2, 0.4]), size=0.03)
            # Draw current swing desired position (intermediate along trajectory) in orange
            debug.draw_world_position(viewer, swing_des, np.array([1.0, 0.6, 0.0, 0.6]), size=0.04)

        return QPControllerResult(
            tau=tau,
            tau_contact=tau_contact,
            contact_forces=contact_forces,
            wrench_des=wrench_des,
            G=G,
        )


# Convenience factory for external use
def create_mpc_controller(
    com_height: float = 0.45,
    horizon_steps: int = 3,
    step_time: float = 0.4,
    v_des: float = 0.5,
) -> ModelPredictiveController:
    lipm = LIPMParams(com_height=com_height)
    planner = FootstepMPCPlanner(step_time=step_time, horizon_steps=horizon_steps, v_des=v_des)
    return ModelPredictiveController(lipm=lipm, planner=planner)


