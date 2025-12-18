"""Task 4: Stair climbing (MPC controller + terrain-aware foot placement).

The stairs are already present in `biped_robot.xml` as geoms named `step0..step4`,
with rise=0.10m and run=0.20m.

Safety rule (optional but recommended for the assignment): only the foot sphere
geoms may contact the step geoms. Any other robot geom contacting a step counts
as a violation (helps ensure no body/leg-edge collisions).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import mujoco
import numpy as np

import sim_runner as sim
from mpc_controller_stair import create_mpc_controller
from terrain_utils import iter_step_geom_ids
import visualization_utils

HERE = os.path.dirname(os.path.abspath(__file__))


class Task4StairController:
    """Adapter: wrap the stair MPC controller to match `sim_runner`'s expected API."""

    def __init__(self, *, v_des: float, step_time: float, horizon_steps: int, swing_height: float) -> None:
        self._mpc = create_mpc_controller(
            com_height=0.45,
            horizon_steps= 10,
            step_time= 0.14,
            v_des= -1.4,
        )
        # Increase swing clearance for stair edges (this affects the arc mid-swing).
        self._mpc.swing_height = .17

        # Video recording
        self.video_frames: list[np.ndarray] = []
        self.renderer: Optional[mujoco.Renderer] = None
        self.recording_enabled = False

    def enable_video_recording(self, model: mujoco.MjModel) -> None:
        try:
            self.renderer = visualization_utils.create_video_renderer(model)
            self.recording_enabled = True
            print(
                f"[INFO] Task4 video recording enabled "
                f"({visualization_utils.VIDEO_WIDTH}x{visualization_utils.VIDEO_HEIGHT} @ {visualization_utils.VIDEO_FPS} fps)"
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize video renderer: {e}")
            self.renderer = None
            self.recording_enabled = False

    def save_video(self, output_dir: Optional[str] = None) -> None:
        visualization_utils.save_video(self.video_frames, output_dir, filename="task4_stair.mp4")

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, t: float):
        qp_result = self._mpc(model=model, data=data, t=t)
        # sim_runner expects either tau or (tau, info). Provide info for debugging.
        tau = np.asarray(qp_result.tau, dtype=float)
        info = {
            "raw_tau": tau.copy(),
            "contact_forces": np.asarray(qp_result.contact_forces, dtype=float).copy(),
        }

        # Video capture (offscreen)
        if self.recording_enabled and self.renderer is not None:
            frame_period = 1.0 / float(visualization_utils.VIDEO_FPS)
            expected_frames = int(float(t) / frame_period)
            if len(self.video_frames) < expected_frames:
                self.renderer.update_scene(data, camera="side_follow")
                frame = self.renderer.render()
                self.video_frames.append(visualization_utils.frame_to_uint8_rgb(frame))

        return tau, info


def _no_nonfoot_step_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau_cmd: np.ndarray,
    info: Optional[dict[str, Any]],
) -> Optional[str]:
    left_foot_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
    right_foot_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
    foot_geoms = {int(left_foot_gid), int(right_foot_gid)}

    step_geoms = set(iter_step_geom_ids(model, name_prefix="step"))
    if not step_geoms:
        return "no step geoms found (expected names like step0..step4)"

    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        if g1 < 0 or g2 < 0:
            continue

        if g1 in step_geoms and g2 not in foot_geoms:
            step_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
            return f"non-foot contact with step: {other_name} touched {step_name}"
        if g2 in step_geoms and g1 not in foot_geoms:
            step_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
            return f"non-foot contact with step: {other_name} touched {step_name}"

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 4: Stair climbing")
    p.add_argument("--interactive", action="store_true", help="launch viewer")
    p.add_argument("--sim-time", type=float, default=12.0)
    p.add_argument("--perturb", action="store_true")
    p.add_argument("--ignore-violations", action="store_true", help="continue even after a violation")
    p.add_argument("--v-des", type=float, default=0.6, help="desired forward speed (m/s)")
    p.add_argument("--step-time", type=float, default=0.40)
    p.add_argument("--horizon-steps", type=int, default=3)
    p.add_argument(
        "--swing-height",
        type=float,
        default=0.20,
        help="minimum swing-foot clearance used for the arc (meters). Try 0.20–0.30 for stairs.",
    )
    p.add_argument(
        "--disable-step-collision-check",
        action="store_true",
        help="do not stop when body/legs touch step geoms (debugging only)",
    )
    p.add_argument("--video-dir", type=str, default=None, help="directory to save MP4 (default: task4_result/videos)")
    p.add_argument("--no-video", action="store_true", help="disable offscreen video recording")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    controller = Task4StairController(
        v_des=args.v_des,
        step_time=args.step_time,
        horizon_steps=args.horizon_steps,
        swing_height=args.swing_height,
    )

    if not args.no_video:
        from sim_runner import XML_PATH

        temp_model = mujoco.MjModel.from_xml_path(XML_PATH)
        controller.enable_video_recording(temp_model)

    result = sim.run_simulation(
        controller,
        sim_time=float(args.sim_time),
        interactive=bool(args.interactive),
        perturb=bool(args.perturb),
        constraint_checker=None if args.disable_step_collision_check else _no_nonfoot_step_contacts,
        description="Task 4 (stair climbing)",
        stop_on_violation=False,
    )

    if not args.no_video:
        out_dir = args.video_dir or os.path.join(HERE, "task4_result", "videos")
        controller.save_video(out_dir)
        print(f"[INFO] Task4 simulation complete for t={result.final_time:.2f}s (video saved in {out_dir})")

if __name__ == "__main__":
    main()
