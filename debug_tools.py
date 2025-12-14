
"""Debug visualization tools for MuJoCo simulations."""
from __future__ import annotations
def clear_arrows(viewer: mujoco.viewer.Handle) -> None:
    """Clear all debug arrows from the viewer (call at start of frame)."""
    viewer.user_scn.ngeom = 0

from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

def _fmt_debug(arr: np.ndarray) -> str:
    """Format numpy arrays for concise debug output."""
    return np.array2string(np.asarray(arr, dtype=float), precision=4, suppress_small=False)



def draw_contact_force_arrows(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: dict[str, Any],
) -> None:
    """Draw contact force vectors as arrows in the viewer.
    
    Args:
        viewer: MuJoCo viewer handle
        model: MuJoCo model
        data: MuJoCo data
        info: Controller info dict containing:
            - contact_forces: [Fx_r, Fy_r, Fx_l, Fy_l]
            - right_contact: bool
            - left_contact: bool
    """
    contact_forces = info.get("contact_forces")
    right_contact = info.get("right_contact", False)
    left_contact = info.get("left_contact", False)
    
    if contact_forces is None:
        return
    
    Fx_r, Fz_r, Fx_l, Fz_l = contact_forces
    
    # Get foot body IDs and positions
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    
    right_pos = data.xpos[right_id].copy()
    left_pos = data.xpos[left_id].copy()
    
    # Scale factor for visualization
    force_scale = 0.003
    
    # Draw right foot force arrow
    if right_contact and (abs(Fx_r) > 0.1 or abs(Fz_r) > 0.1):
        # Force vector in world frame (X=forward, Y=vertical, Z=lateral)
        force_r = np.array([Fx_r, 0.0, Fz_r])
        force_end_r = right_pos + force_scale * force_r
        
        viewer.user_scn.ngeom += 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array([1.0, 0.0, 0.0, 0.6]),  # red
        )
        mujoco.mjv_makeConnector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.01,
            right_pos[0], right_pos[1], right_pos[2],
            force_end_r[0], force_end_r[1], force_end_r[2],
        )
    
    # Draw left foot force arrow
    if left_contact and (abs(Fx_l) > 0.1 or abs(Fz_l) > 0.1):
        force_l = np.array([Fx_l, 0.0, Fz_l])
        force_end_l = left_pos + force_scale * force_l
        
        viewer.user_scn.ngeom += 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array([0.0, 0.0, 1.0, 0.6]),  # blue
        )
        mujoco.mjv_makeConnector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.01,
            left_pos[0], left_pos[1], left_pos[2],
            force_end_l[0], force_end_l[1], force_end_l[2],
        )

def draw_arrow(viewer: mujoco.viewer.Handle, start: np.ndarray, end: np.ndarray, color: np.ndarray) -> None:
    """Draw a single arrow in the MuJoCo viewer.
    
    Args:
        viewer: MuJoCo viewer handle
        start: 3D start position of the arrow
        end: 3D end position of the arrow
        color: RGBA color of the arrow
    """
    viewer.user_scn.ngeom += 1
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        color,
    )
    mujoco.mjv_makeConnector(
        viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.01,
        start[0], start[1], start[2],
        end[0], end[1], end[2],
    )

def draw_world_position(
    viewer: mujoco.viewer.Handle,
    position: np.ndarray,
    color: np.ndarray,
    size: float = 0.06,
) -> None:
    """Draw a dot at the world position"""
    viewer.user_scn.ngeom += 1
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0.0, 0.0]),
        position,
        np.eye(3).flatten(),
        color,
    )

def draw_tau(
    viewer: mujoco.viewer.Handle,
    data: mujoco.MjData,
    # sim: Any,
    tau: np.ndarray,
) -> None:
    """Draw joint torque arrows at the joints in the viewer.
    
    Args:
        viewer: MuJoCo viewer handle
        data: MuJoCo data
        sim: Simulation object with TORQUE_LIMITS attribute
        tau: Joint torques to visualize
    """
    joint_ids = [3, 4, 6, 7]
    for i in range(4):
        joint_pos = data.xpos[joint_ids[i]]
        draw_arrow(
            viewer,
            joint_pos,
            joint_pos + np.array([0.0, tau[i] * 0.1, 0.0]),
            np.array([0.0, 0.4, 1.0, 0.6])
        )
