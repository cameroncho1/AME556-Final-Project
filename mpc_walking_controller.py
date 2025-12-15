"""MPC-based footstep planner and controller for biped walking.
Controls swing foot with PD, stance foot with QP.
"""
import numpy as np
import mujoco
from sim_runner import TrunkState, foot_contacts, get_trunk_state, MU, TORQUE_LIMITS
import debug_tools as debug
from qp_solver import qp_controller, QPControllerResult
import sim_runner as sim
# import visualization_utils as viz
# PD gains for swing foot
# Kp_swing_x = 100
Kp_swing_x = 0
Kd_swing_x = 0
# Kp_swing_z = 100
Kp_swing_z = 0
Kd_swing_z = 0


# --- Simple horizon-based MPC for footstep planning ---
class MPCFootstepPlanner:
    def __init__(self, step_time=0.5, horizon=5, step_length_nom=0.25, v_des=0.5):
        self.step_time = step_time
        self.horizon = horizon
        self.step_length_nom = step_length_nom
        self.v_des = v_des
        self.reset()

    def reset(self):
        self.t = 0.0
        self.foot = 'left'  # start with left foot swing
        self.next_switch = self.step_time
        self.footsteps = []

    def plan(self, trunk_pos, trunk_vel, direction=1):
        # Predict trunk x over horizon using current velocity
        x0 = trunk_pos[0]
        v0 = trunk_vel[0]
        N = self.horizon
        dt = self.step_time
        v_des = self.v_des * direction
        # Decision variables: footstep locations f[0],...,f[N-1]
        # Cost: sum (f_k - (x0 + v0*(k+1)*dt))^2 + alpha*(f_k - f_{k-1} - step_length_nom)^2
        alpha = 1.0
        f = np.zeros(N)
        f_prev = x0 - self.step_length_nom * direction  # previous stance foot
        for k in range(N):
            # Predict where trunk will be at step k using current velocity
            xk = x0 + v_des * (k+1) * dt
            # Optionally blend with desired velocity for robustness
            # xk = x0 + ((1-beta)*v0 + beta*v_des) * (k+1) * dt  # beta in [0,1]
            # Regularize step length
            if k == 0:
                reg = (f_prev + self.step_length_nom * direction)
            else:
                reg = (f[k-1] + self.step_length_nom * direction)
            # Weighted sum
            f[k] = (xk + alpha * reg) / (1 + alpha)
        footsteps = [np.array([fx, 0.0, 0.0]) for fx in f]
        self.footsteps = footsteps
        return footsteps

    def update(self, t):
        self.t = t
        if t >= self.next_switch:
            self.foot = 'right' if self.foot == 'left' else 'left'
            self.next_switch += self.step_time

    def get_swing_foot(self):
        return self.foot

    def get_next_footstep(self):
        if self.footsteps:
            return self.footsteps.pop(0)
        return None


def mpc_walking_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    planner: MPCFootstepPlanner,
    t: float,
    pos_des: np.ndarray,
    vel_des: np.ndarray,
    theta_des: float,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    direction: int = 1
) -> QPControllerResult:
    """True horizon-based MPC walking controller: plans footsteps over horizon, controls swing with PD, stance with QP."""
    # Update planner and get swing/stance
    planner.update(t)
    swing_foot = planner.get_swing_foot()
    contact_left, contact_right = foot_contacts(model, data)

    # Plan footsteps using horizon-based MPC
    trunk_state = get_trunk_state(model, data, "body_frame")
    if not planner.footsteps:
        planner.plan(trunk_state.pos, trunk_state.vel, direction)
    next_footstep = planner.get_next_footstep()

    # Desired foot positions
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    right_pos = data.xpos[right_id].copy()
    left_pos = data.xpos[left_id].copy()

    # Set desired position for swing foot
    swing_des = next_footstep if next_footstep is not None else (left_pos if swing_foot=='left' else right_pos)
    if swing_foot == 'left':
        # Left foot swings, right is stance
        stance_contact = contact_right
        swing_contact = contact_left
        swing_curr = left_pos
        stance_curr = right_pos
        swing_idx = 2  # left foot
    else:
        stance_contact = contact_left
        swing_contact = contact_right
        swing_curr = right_pos
        stance_curr = left_pos
        swing_idx = 0  # right foot

    swing_des[2] = 0.4
    #calcualte the q_des
    # q_des_swing = q_des.copy()
    # if swing_foot == 'left':
    #     joint_indices = [model.joint_name2id('hip_L'), model.joint_name2id('knee_L')]
    # else:
    #     joint_indices = [model.joint_name2id('hip_R'), model.joint_name2id('knee_R')]
    # q_init = data.qpos[joint_indices].copy()
    # Inverse kinematics for swing foot
    # q_des_swing = solve_ik(model, data, swing_foot, swing_des, q_init)
    # If swing foot is in contact, use QP for all joints (dual support)
    
    # if swing_contact:
    #     qp_result = qp_controller(
    #         model, data, pos_des, vel_des, theta_des, q_des, qd_des
    #     )
    #     tau = qp_result.tau.copy()
    # else:
    if True:
        # QP for stance, PD for swing
        qp_result = qp_controller(
            model, data, pos_des, vel_des, theta_des, q_des, qd_des, 
            left_enable=(swing_foot != 'left'),
            right_enable=(swing_foot != 'right')
            # left_enable=True,
            # right_enable=True
        )
        tau = qp_result.tau.copy()
        # tau = np.zeros(4)  # 4 joints: [hip_R, knee_R, hip_L, knee_L]
        # PD in world frame for swing foot
        swing_err = swing_des - swing_curr
        swing_des_c = swing_des.copy()
        swing_des_c[2] = .2 * abs(swing_err[0])
        swing_err_c = swing_des_c - swing_curr
        swing_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, swing_foot+"_foot")
        swing_vel = data.cvel[swing_id][3:]
        # print(f"[DEBUG][MPC] swing_foot: {swing_foot}, swing_des: {swing_des}, swing_curr: {swing_curr}, swing_err: {swing_err}, swing_vel: {swing_vel}")
        # print(f"[DEBUG][MPC] swing_err_c: {swing_err_c}")
        # print(f"[DEBUG][MPC] swing_vel: {swing_vel}")
        # print(f"[DEBUG][MPC] before swing tau: {tau}")
        # print(f"[DEBUG][MPC] qp_result.tau: {qp_result.tau}")
        # swing_force = (Kp_swing * swing_err_c - Kd_swing * swing_vel)
        swing_force_x = (Kp_swing_x * swing_err_c[0] - Kd_swing_x * swing_vel[0])
        swing_force_z = (Kp_swing_z * swing_err_c[2] - Kd_swing_z * swing_vel[2])
        debug.draw_arrow(
            sim.get_viewer(), swing_curr,
            swing_curr + np.array([-swing_force_x, 0.0, -swing_force_z]),
            np.array([1.0,0.5,0.0,0.7])
        )
        
        # Map to joint torques (simple Jacobian transpose)
    # Compute foot Jacobians for QP
        jacp_r = np.zeros((3, model.nv))
        jacp_l = np.zeros((3, model.nv))
        jacr_tmp = np.zeros((3, model.nv))
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        mujoco.mj_jacBody(model, data, jacp_r, jacr_tmp, right_id)
        mujoco.mj_jacBody(model, data, jacp_l, jacr_tmp, left_id)
        Jr = np.vstack([jacp_r[0, 3:7], jacp_r[2, 3:7]])
        Jl = np.vstack([jacp_l[0, 3:7], jacp_l[2, 3:7]])

        swing_body_id = left_id if swing_foot=='left' else right_id
        # mujoco.mj_jacBody(model, data, jacp, jacr_tmp, swing_body_id)
        swing_foot_pos = data.xpos[swing_body_id].copy()
        debug.draw_world_position(sim.get_viewer(), swing_foot_pos, np.array([1.0,1.0,0.0,0.7]))
        # debug.draw_world_position(sim.get_viewer(), swing_des, np.array([0.0,1.0,1.0,0.7]))
        # Assign swing torques to correct joint indices
        if swing_foot == 'left':
            # tau[0:2] += jacp[:2,0:2].T @ swing_force[:2]  # left hip, knee
            tau += Jl.T @ np.array([-swing_force_x, -swing_force_z])  # left hip, knee    
            print(f"[DEBUG][MPC] left tau after PD swing: {tau}")
            pass
        
        elif swing_foot == 'right':
            # tau[2:4] += jacp[:2,2:4].T @ swing_force[:2]  # right hip, knee
            tau += Jr.T @ np.array([-swing_force_x, -swing_force_z])  # right hip, knee
            print(f"[DEBUG][MPC] right tau after PD swing: {tau}")
            pass
        # Debug draw
        # debug.draw_arrow(
        #     sim.get_viewer(), swing_curr, swing_des, np.array([1.0,0.5,0.0,0.7])
        # )

    # Debug draw footsteps
    # for f in planner.footsteps:
    #     debug.draw_world_position(sim.get_viewer(), f, np.array([0.0,1.0,0.0,0.3]))
    debug.draw_world_position(sim.get_viewer(), swing_des_c, np.array([1.0,0.0,0.0,0.7]))

    # Clip to torque limits
    tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

    debug.draw_tau(
        sim.get_viewer(),
        data,
        tau
    )

    return QPControllerResult(
        tau,
        tau_contact=qp_result.tau_contact,
        contact_forces=qp_result.contact_forces,
        wrench_des=qp_result.wrench_des,
        G=qp_result.G
    )
