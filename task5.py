"""Task 5: Obstacle course.

Runs the MPC controller over a simple obstacle course terrain defined in
`biped_robot_task5.xml`.

Collision policy:
- Feet may contact the ground plane.
- By default, **no robot geom may contact obstacle geoms** (names starting with `obs`).
  You can optionally allow feet to touch obstacles with `--allow-feet-on-obstacles`.
- Optionally enforce "no non-foot contacts with ground" (prevents body/leg scraping).

Video:
- Saves MP4 to `task5_result/videos/task5_obstacle_course.mp4` by default.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import mujoco
import numpy as np

import sim_runner as sim
import visualization_utils
from mpc_controller_stair import create_mpc_controller

HERE = os.path.dirname(os.path.abspath(__file__))
XML_TASK5 = os.path.join(HERE, "biped_robot_task5.xml")


class Task5Controller:
    """Adapter wrapper that also records video frames."""

    def __init__(self, *, v_des: float, step_time: float, horizon_steps: int, swing_height: float) -> None:
        self._mpc = create_mpc_controller(
            com_height=0.45,
            horizon_steps=int(horizon_steps),
            step_time=float(step_time),
            v_des=float(v_des),
        )
        self._mpc.swing_height = float(swing_height)

        self.video_frames: list[np.ndarray] = []
        self.renderer: Optional[mujoco.Renderer] = None
        self.recording_enabled = False

    def enable_video_recording(self, model: mujoco.MjModel) -> None:
        try:
            self.renderer = visualization_utils.create_video_renderer(model)
            self.recording_enabled = True
            print(
                f"[INFO] Task5 video recording enabled "
                f"({visualization_utils.VIDEO_WIDTH}x{visualization_utils.VIDEO_HEIGHT} @ {visualization_utils.VIDEO_FPS} fps)"
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize video renderer: {e}")
            self.renderer = None
            self.recording_enabled = False

    def save_video(self, output_dir: Optional[str]) -> None:
        visualization_utils.save_video(
            self.video_frames,
            output_dir,
            filename="task5_obstacle_course.mp4",
        )

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, t: float):
        qp_result = self._mpc(model=model, data=data, t=t)
        tau = np.asarray(qp_result.tau, dtype=float)
        info = {
            "raw_tau": tau.copy(),
            "contact_forces": np.asarray(qp_result.contact_forces, dtype=float).copy(),
        }

        if self.recording_enabled and self.renderer is not None:
            frame_period = 1.0 / float(visualization_utils.VIDEO_FPS)
            expected_frames = int(float(t) / frame_period)
            if len(self.video_frames) < expected_frames:
                self.renderer.update_scene(data, camera="side_follow")
                frame = self.renderer.render()
                self.video_frames.append(visualization_utils.frame_to_uint8_rgb(frame))

        return tau, info


def _collision_checker_factory(
    *,
    forbid_nonfoot_terrain_contacts: bool,
) -> sim.ConstraintFn:
    def _checker(model: mujoco.MjModel, data: mujoco.MjData, tau_cmd: np.ndarray, info: Optional[dict[str, Any]]):
        left_foot_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom"))
        right_foot_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom"))
        foot_geoms = {left_foot_gid, right_foot_gid}
        terrain_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground_body"))

        for i in range(int(data.ncon)):
            con = data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if g1 < 0 or g2 < 0:
                continue

            # Terrain/obstacle contact policy:
            # Any geom under `ground_body` is considered terrain/obstacle.
            # Only feet are allowed to touch it (when enabled).
            if forbid_nonfoot_terrain_contacts:
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
                if b1 == terrain_body and g2 not in foot_geoms:
                    terr_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
                    other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
                    return f"non-foot contacted terrain: {other_name} touched {terr_name}"
                if b2 == terrain_body and g1 not in foot_geoms:
                    terr_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
                    other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
                    return f"non-foot contacted terrain: {other_name} touched {terr_name}"

        return None

    return _checker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 5: Obstacle course")
    p.add_argument("--interactive", action="store_true", help="launch viewer")
    p.add_argument("--sim-time", type=float, default=14.0)
    p.add_argument("--perturb", action="store_true")
    p.add_argument("--ignore-violations", action="store_true", help="continue even after a violation")

    p.add_argument("--v-des", type=float, default=0.7, help="desired forward speed (m/s)")
    p.add_argument("--step-time", type=float, default=0.35)
    p.add_argument("--horizon-steps", type=int, default=4)
    p.add_argument("--swing-height", type=float, default=0.25, help="swing clearance arc (m)")

    p.add_argument(
        "--forbid-nonfoot-terrain-contacts",
        action="store_true",
        help="treat any non-foot contact with any `ground_body` geom as a violation (terrain + obstacles)",
    )

    p.add_argument("--video-dir", type=str, default=None, help="directory to save MP4 (default: task5_result/videos)")
    p.add_argument("--no-video", action="store_true", help="disable offscreen video recording")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Point sim_runner at the task-specific XML without changing global defaults elsewhere.
    sim.XML_PATH = XML_TASK5

    controller = Task5Controller(
        v_des=args.v_des,
        step_time=args.step_time,
        horizon_steps=args.horizon_steps,
        swing_height=args.swing_height,
    )

    if not args.no_video:
        temp_model = mujoco.MjModel.from_xml_path(sim.XML_PATH)
        controller.enable_video_recording(temp_model)

    checker = _collision_checker_factory(forbid_nonfoot_terrain_contacts=bool(args.forbid_nonfoot_terrain_contacts))

    result = sim.run_simulation(
        controller,
        sim_time=float(args.sim_time),
        interactive=bool(args.interactive),
        perturb=bool(args.perturb),
        constraint_checker=checker,
        description="Task 5 (obstacle course)",
        stop_on_violation=False,
    )

    if not args.no_video:
        out_dir = args.video_dir or os.path.join(HERE, "task5_result", "videos")
        controller.save_video(out_dir)
        print(f"[INFO] Task5 simulation complete for t={result.final_time:.2f}s (video saved in {out_dir})")


if __name__ == "__main__":
    main()


