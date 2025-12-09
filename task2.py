"""Task 2 standing controller using QP-based contact wrench allocation."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from sim_runner import TORQUE_LIMITS, run_simulation

try:
    from cvxopt import matrix, solvers  # type: ignore

    solvers.options["show_progress"] = False
    HAS_CVXOPT = True
except Exception:
    HAS_CVXOPT = False

MASS_TOTAL = 9.0  # 8 kg body + 4x0.25 kg legs
GRAVITY = 9.81
MU = 0.7
FY_MIN = 10.0
FY_MAX = 250.0

Kp_root_xy = 60.0
Kd_root_xy = 12.0
Kp_root_theta = 80.0
Kd_root_theta = 15.0
Kp_joint = 120.0
Kd_joint = 25.0


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
    # floor_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plane")
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
        if right_body in bodies:
            right = True
        if left_body in bodies:
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

    v = np.array([0.0, 1.0, 0.0, 1.0])
    weight = 0.1
    mg = MASS_TOTAL * GRAVITY
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
        h_list = [0.0, 0.0, 0.0, 0.0, -FY_MIN, FY_MAX, -FY_MIN, FY_MAX]
        G = matrix(np.array(G_list, dtype=float))
        h = matrix(np.array(h_list, dtype=float))
        try:
            sol = solvers.qp(P, q, G, h)
            F = np.array(sol["x"]).flatten()
            print("Force result from QP:", F)
            return F
        except Exception:
            pass

    # Least-squares fallback
    F_ls = np.linalg.pinv(A) @ tau_des
    Fx_r, Fy_r, Fx_l, Fy_l = F_ls
    Fy_r = np.clip(max(Fy_r, mg * 0.5), FY_MIN, FY_MAX)
    Fy_l = np.clip(max(Fy_l, mg * 0.5), FY_MIN, FY_MAX)
    Fx_r = np.clip(Fx_r, -MU * Fy_r, MU * Fy_r)
    Fx_l = np.clip(Fx_l, -MU * Fy_l, MU * Fy_l)
    return np.array([Fx_r, Fy_r, Fx_l, Fy_l])


class StandingQPController:
    def __init__(self, profile: HeightProfile):
        self.profile = profile

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, t: float):
        # print("Running QP standing controller at time {:.2f}s".format(t))
        if not both_feet_in_contact(model, data):
            tau = np.zeros(4)
            return tau, {"raw_tau": tau.copy()}
        print("Both feet in contact at time {:.2f}s".format(t))
        x = data.qpos[0]
        y = data.qpos[1]
        theta = data.qpos[2]
        xd = data.qvel[0]
        yd = data.qvel[1]
        thetad = data.qvel[2]

        q = data.qpos[3:7]
        qd = data.qvel[3:7]

        y_des, yd_des = self.profile.desired_height(t)
        x_des = 0.0
        theta_des = 0.0

        fx_des = Kp_root_xy * (x_des - x) + Kd_root_xy * (0.0 - xd)
        fy_des = Kp_root_xy * (y_des - y) + Kd_root_xy * (yd_des - yd)
        tau_theta = Kp_root_theta * (theta_des - theta) + Kd_root_theta * (0.0 - thetad)

        q1_des = -math.pi / 3
        q2_des = math.pi / 2
        q3_des = -math.pi / 6
        q4_des = math.pi / 2
        q_des = np.array([q1_des, q2_des, q3_des, q4_des])
        tau_posture = Kp_joint * (q_des - q) - Kd_joint * qd

        tau_bias = data.qfrc_bias[3:7].copy()
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_frame")
        mujoco.mj_jacBody(model, data, jacp, jacr, trunk_id)
        J_trunk = np.zeros((3, 4))
        J_trunk[0, :] = jacp[0, 3:7]
        J_trunk[1, :] = jacp[2, 3:7]
        J_trunk[2, :] = jacr[1, 3:7]
        tau_root = J_trunk.T @ np.array([fx_des, fy_des, tau_theta])
        tau_des = tau_root + tau_posture - tau_bias

        jacp_r = np.zeros((3, model.nv))
        jacp_l = np.zeros((3, model.nv))
        jacr_tmp = np.zeros((3, model.nv))
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shin")
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shin")
        mujoco.mj_jacBody(model, data, jacp_r, jacr_tmp, right_id)
        mujoco.mj_jacBody(model, data, jacp_l, jacr_tmp, left_id)
        Jr = np.vstack([jacp_r[0, 3:7], jacp_r[2, 3:7]])
        Jl = np.vstack([jacp_l[0, 3:7], jacp_l[2, 3:7]])
        # print("Solving QP for contact forces...")
        F = solve_contact_qp(Jr, Jl, tau_des)
        Fx_r, Fy_r, Fx_l, Fy_l = F
        tau = Jr.T @ np.array([Fx_r, Fy_r]) + Jl.T @ np.array([Fx_l, Fy_l])
        tau += 0.02 * tau_posture
        tau_raw = tau.copy()
        tau = np.clip(tau_raw, -150.0, 150.0)
        return tau, {"raw_tau": tau_raw}


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = StandingQPController(HeightProfile())
    run_simulation(
        controller,
        sim_time=args.sim_time,
        interactive=args.interactive,
        perturb=args.perturb,
        description="Task 2 QP standing",
        stop_on_violation=not args.ignore_violations,
    )


if __name__ == "__main__":
    main()
