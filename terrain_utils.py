"""
Utilities for interacting with terrain geometry (e.g., stairs) in MuJoCo.

This module is intentionally small: it provides
- `raycast_height_at_xy`: query the terrain surface height at (x, y) via a vertical ray.
- `project_xy_away_from_edges`: adjust a candidate (x, y) away from nearby "drop" edges,
  and return the resulting (x, y) along with the surface height.

These utilities are used by `mpc_controller stair.py` to place feet on top of stair treads
and avoid landing too close to stair edges (reducing leg/body collisions with edges).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import mujoco
import numpy as np


@dataclass(frozen=True)
class StaircaseSpec:
    """Axis-aligned staircase made of stacked boxes inside `ground_body`.

    Coordinate convention:
    - Stairs run along +X
    - Width spans Y (centered at `y_center`)
    - Ground plane is at z=0

    Each step i (0-indexed) has:
    - riser height = `rise`
    - tread length = `run`
    - top surface at z = (i+1)*rise
    - center at:
        x = x_start + run/2 + i*run
        y = y_center
        z = (i+0.5)*rise
    - box half-sizes:
        sx = run/2, sy = width/2, sz = rise/2
    """

    n_steps: int = 5
    rise: float = 0.10  # meters
    run: float = 0.20  # meters
    width: float = 2.0  # meters (total width in Y)
    x_start: float = 1.0  # meters (front face of first riser is at x_start)
    y_center: float = 0.0
    name_prefix: str = "step"
    rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

    def step_geom_name(self, i: int) -> str:
        return f"{self.name_prefix}{i}"


def _terrain_body_id(model: mujoco.MjModel, terrain_body: str) -> int:
    """Resolve the body id for the terrain container body (default: 'ground_body')."""
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, terrain_body))


def raycast_height_at_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    *,
    z_from: float = 2.0,
    bodyexclude: int = -1,
    terrain_body: str = "ground_body",
    min_z: float = 0.0,
    max_tries: int = 20,
) -> Tuple[float, int]:
    """
    Cast a vertical ray downwards from (x, y, z_from) and return:
    - surface z at intersection
    - geom id hit (or -1 if no hit)

    Notes:
    - We only accept intersections whose geom belongs to `terrain_body`. This avoids
      accidentally returning the robot's own geometry when raycasting near the feet.
    - If the first hit is not terrain, we step the ray origin slightly below that hit
      and try again (up to `max_tries`).
    """
    ground_bid = _terrain_body_id(model, terrain_body)

    # Direction: straight down in world frame.
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # MuJoCo writes the hit geom id into this array.
    geomid = np.array([-1], dtype=np.int32)

    # Start from the requested height; step downward if we hit non-terrain.
    pnt = np.array([float(x), float(y), float(z_from)], dtype=np.float64)

    for _ in range(max_tries):
        geomid[0] = -1
        dist = float(mujoco.mj_ray(model, data, pnt, vec, None, 1, int(bodyexclude), geomid))
        if dist < 0.0 or int(geomid[0]) < 0:
            return float(min_z), -1

        gid = int(geomid[0])
        z_hit = float(pnt[2] - dist)  # since vec = (0,0,-1)

        # Accept only terrain geoms (under ground_body).
        if int(model.geom_bodyid[gid]) == ground_bid:
            return z_hit, gid

        # Otherwise, step just below this intersection and try again.
        pnt[2] = z_hit - 1e-3
        if pnt[2] < min_z - 1.0:
            break

    return float(min_z), -1


def raycast_hitpos_at_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    *,
    z_from: float = 2.0,
    bodyexclude: int = -1,
    terrain_body: str = "ground_body",
    min_z: float = 0.0,
    max_tries: int = 20,
) -> Tuple[np.ndarray, int]:
    """Like `raycast_height_at_xy`, but returns the full hit position (x,y,z) and geom id."""
    z, gid = raycast_height_at_xy(
        model,
        data,
        x,
        y,
        z_from=z_from,
        bodyexclude=bodyexclude,
        terrain_body=terrain_body,
        min_z=min_z,
        max_tries=max_tries,
    )
    return np.array([float(x), float(y), float(z)], dtype=np.float64), int(gid)


def project_xy_away_from_edges(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    *,
    z_from: float = 2.0,
    bodyexclude: int = -1,
    terrain_body: str = "ground_body",
    probe: float = 0.04,
    drop_thresh: float = 0.04,
    max_iters: int = 12,
    step_scale: float = 0.6,
) -> Tuple[float, float, float, float]:
    """
    Heuristically move (x, y) away from nearby edges where the terrain height drops.

    We query the terrain height at the candidate point and at a small set of offset
    probe points. If any probe direction sees a drop larger than `drop_thresh`, we
    "push" the candidate away from that direction and iterate.

    Returns (x_proj, y_proj, z_surface_at_proj, edge_score).
    `edge_score` is the maximum observed drop relative to the local surface at the
    final location (0 means "no large nearby drops detected").
    """
    probe = float(max(1e-4, probe))
    drop_thresh = float(max(1e-6, drop_thresh))

    xr = float(x)
    yr = float(y)

    # Use 8-direction probes for robustness (stairs edges are mostly along X, but keep general).
    dirs = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    # Normalize diagonals so the probe radius is consistent.
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    edge_score = 0.0

    for _ in range(int(max_iters)):
        z0, _ = raycast_height_at_xy(
            model, data, xr, yr, z_from=z_from, bodyexclude=bodyexclude, terrain_body=terrain_body, min_z=0.0
        )

        push = np.zeros(2, dtype=np.float64)
        worst_drop = 0.0

        for dxy in dirs:
            xp = xr + probe * float(dxy[0])
            yp = yr + probe * float(dxy[1])
            zp, _ = raycast_height_at_xy(
                model, data, xp, yp, z_from=z_from, bodyexclude=bodyexclude, terrain_body=terrain_body, min_z=0.0
            )
            drop = float(z0 - zp)
            worst_drop = max(worst_drop, drop)
            if drop > drop_thresh:
                # If terrain drops in direction dxy, push opposite that direction.
                push -= dxy * (drop - drop_thresh)

        edge_score = float(max(edge_score, worst_drop))

        norm = float(np.linalg.norm(push))
        if norm < 1e-9:
            break

        # Move a fraction of the probe distance in the push direction.
        push_dir = push / norm
        xr += float(step_scale * probe) * float(push_dir[0])
        yr += float(step_scale * probe) * float(push_dir[1])

    z_final, _ = raycast_height_at_xy(
        model, data, xr, yr, z_from=z_from, bodyexclude=bodyexclude, terrain_body=terrain_body, min_z=0.0
    )
    return float(xr), float(yr), float(z_final), float(edge_score)


def iter_step_geom_ids(model: mujoco.MjModel, *, name_prefix: str = "step") -> Iterable[int]:
    """Yield geom ids whose name starts with `name_prefix` (e.g., step0..step4)."""
    for gid in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.startswith(name_prefix):
            yield int(gid)


def staircase_surface_height_at_x(
    x: float,
    *,
    spec: StaircaseSpec = StaircaseSpec(),
    ground_z: float = 0.0,
) -> float:
    """Analytic height of an ideal staircase at world-x, for quick checks/debug.

    This does *not* query MuJoCo geometry; use `raycast_height_at_xy` for the real surface.
    """
    x_rel = float(x) - float(spec.x_start)
    if x_rel < 0.0:
        return float(ground_z)
    idx = int(np.floor(x_rel / float(spec.run)))
    idx = int(np.clip(idx, 0, int(spec.n_steps) - 1))
    return float((idx + 1) * float(spec.rise))


def get_step_treads_from_model(
    model: mujoco.MjModel,
    *,
    name_prefix: str = "step",
) -> list[dict[str, float]]:
    """Extract axis-aligned tread AABBs for box geoms named like `step0..stepN`.

    Returns a list of dicts with keys: x_min, x_max, y_min, y_max, z_top.
    Assumes step geoms are axis-aligned boxes (no rotation), which matches our XML.
    """
    treads: list[dict[str, float]] = []
    for gid in iter_step_geom_ids(model, name_prefix=name_prefix):
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        pos = model.geom_pos[gid]
        half = model.geom_size[gid]
        x_min = float(pos[0] - half[0])
        x_max = float(pos[0] + half[0])
        y_min = float(pos[1] - half[1])
        y_max = float(pos[1] + half[1])
        z_top = float(pos[2] + half[2])
        treads.append({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_top": z_top})
    # Sort front-to-back for convenience.
    treads.sort(key=lambda d: (d["x_min"], d["z_top"]))
    return treads


