# AME556 Final Project Report

## Group Information
**Group Number:** 8  
**Team Members:**  
- Takatoshi Soeda (soeda@usc.edu, USC ID)  
- Cameron Cho (chocamer@usc.edu , 1499538650)

**Simulation Video:**  
https://link-to-video

**Code Repository:**  
https://github.com/cameroncho1/AME556-Final-Project


## 1. Task 1: Simulation and Physical Constraints

### Approach

Task 1 establishes a constraint-aware MuJoCo simulation framework and verifies that the robot respects all physical limits during simulation.  
A deliberately simple controller is used so that correctness of the simulation and constraint enforcement can be isolated from control complexity.

Two controller modes are supported:

- **Zero-torque mode:** Applies no actuation to validate baseline dynamics and constraint monitoring.
- **PD mode:** Regulates joints toward a nominal posture using a PD controller, followed by explicit torque saturation.

At every simulation step, joint angle limits, joint velocity limits, and actuator torque limits are checked using a centralized constraint checker.  
The simulation can either terminate immediately on violation (grading mode) or continue while logging violations (debug mode).

### Code Structure

#### `task1.py`

**Task1Controller**

- `__init__(mode)`  
  Selects either `"pd"` or `"zero"` control mode.

- `__call__(model, data, t)`  
  Reads joint positions and velocities, computes raw PD torques if enabled, applies torque saturation, and returns `(tau, info)` where `info["raw_tau"]` is used for constraint checking.

**Main Entry Point**

- Parses command-line arguments (interactive viewer, perturbation, ignore violations).
- Launches the simulation via `run_simulation(...)`.

#### `sim_runner.py` (shared across all tasks)

- `run_simulation(...)`  
  Shared MuJoCo simulation loop that handles reset, control invocation, constraint enforcement, and optional visualization.

- `default_constraint_checker(...)`  
  Enforces joint angle, joint velocity, and torque limits.

- `default_reset(...)`  
  Resets the robot to the XML keyframe with optional joint perturbations.

- `foot_contacts(...)`  
  Detects left and right foot ground contact.

- `get_trunk_state(...)`  
  Extracts trunk position, orientation, and velocities from the simulation.

### Plots (Task 1)

![Hip joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task1/hip_q_qd_tau_vs_time.png)

**Figure 1.** Hip joint angles, velocities, and torques over time for the Task 1 PD
controller. All values remain within specified physical limits.

![Knee joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task1/knee_q_qd_tau_vs_time.png)


**Figure 2.** Knee joint angles, velocities, and torques over time for the Task 1 PD
controller. Actuator torques remain within saturation limits and joint constraints
are respected throughout the simulation.

### Simulation video link (Task 1):
https://drive.google.com/file/d/1sO9Qa1a86yBy1ZxL9qIJz0DnCTPPp47L/view?usp=sharing

## 2. Task 2: Standing Control with QP-Based Contact Force Allocation

### Approach

Task 2 implements a QP-based standing controller that maintains upright posture while tracking a time-varying trunk height trajectory.

At each control step:

1. A desired trunk height and vertical velocity are generated using a piecewise trajectory profile.
2. A desired trunk wrench \([f_x, f_z, \tau_\theta]\) is computed using PD control with gravity compensation.
3. A grasp matrix maps foot contact forces to the desired trunk wrench.
4. A quadratic program allocates contact forces while enforcing friction cone and vertical force constraints.
5. Contact forces are mapped to joint torques using foot Jacobians.
6. Posture regularization and bias torques are added.
7. Final torques are clipped to actuator limits and applied.

### Code Structure

#### `task2.py`

**HeightProfile**

- `desired_height(t)`  
  Returns desired trunk height and vertical velocity using a hold → rise → drop → hold trajectory.

**StandingQPController**

- `__init__(...)`  
  Initializes trajectory profile, logging buffers, and video recording state.

- `enable_video_recording(model)`  
  Sets up offscreen rendering for video capture.

- `__call__(model, data, t)`  
  Computes desired trunk motion, calls the QP controller, logs states and forces, and optionally records video frames.

- `save_plots(...)`  
  Saves diagnostic plots.

- `save_video(...)`  
  Writes MP4 video from recorded frames.

**Main Entry**

- Parses CLI arguments.
- Runs the simulation via `run_simulation(...)`.

#### `qp_solver.py`

- `solve_contact_qp(G, wrench_des, contact_left, contact_right)`  
  Solves contact force allocation using a CVXOPT QP when available, with a least-squares fallback.  
  Handles left-only, right-only, and double-support cases while enforcing friction and vertical force limits.

- `qp_controller(...)`  
  Computes trunk PD wrench, builds the grasp matrix, solves the QP, converts forces to joint torques, adds posture regularization and bias torques, clips to actuator limits, and returns `QPControllerResult`.

**QPControllerResult**

- `tau` – final applied joint torques  
- `tau_contact` – torques due to contact forces only  
- `contact_forces` – \([F_{x,r}, F_{z,r}, F_{x,l}, F_{z,l}]\)  
- `wrench_des` – desired trunk wrench  
- `G` – grasp matrix  

#### `visualization_utils.py`

- `save_standing_controller_plots(...)`  
  Generates plots for trunk tracking, joint states, torques vs. limits, and contact forces vs. bounds.

- `save_video(...)`  
  Writes MP4 video.

- `create_video_renderer(...)`, `frame_to_uint8_rgb(...)`  
  Utilities for consistent video rendering.

### Plots (Task 2 – Standing)

![Hip joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2/hip_q_qd_tau_vs_time.png)

**Figure 3.** Hip joint angles, velocities, and torques over time for the Task 2

![Knee joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2/knee_q_qd_tau_vs_time.png)

**Figure 4.** Knee joint angles, velocities, and torques over time for the Task 2

![Task 2 trunk state tracking](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2/task2_state_tracking.png)

**Figure 5.** Trunk vertical position and pitch angle tracking over time for the Task 2 standing controller. The controller maintains upright posture while following the desired height trajectory.

![Task 2 joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2/task2_joint_states.png)

**Figure 6.** Joint angles, velocities, and commanded torques over time for the Task 2 standing controller. 

### Simulation video link (Task 2):
https://drive.google.com/file/d/1vaC-tNPepqjLgZ5WdEXffF0YpFKvbGf0/view?usp=sharing

## 3. Task 2 (Walking): MPC Footstep Planning and QP Contact Allocation

### Approach

The walking controller extends the standing framework by incorporating model-predictive trunk control and footstep planning to achieve forward locomotion.

At each control step:

1. A short-horizon sequence of alternating foot placements is generated using capture-point–inspired logic.
2. Trunk dynamics are predicted over the horizon using a Linear Inverted Pendulum Model (LIPM).
3. A desired trunk wrench is computed using LIPM feedforward terms combined with PD feedback.
4. The wrench is allocated to foot contact forces using the shared QP solver.
5. Contact forces are mapped to joint torques via Jacobians.
6. Swing-leg motion is generated using a latched target, parabolic height profile, and planar inverse kinematics.
7. Joint-space posture stabilization is applied and torques are clipped to actuator limits.

### Code Structure

#### `mpc_controller.py`

**FootstepPlan**

- `first_target()`  
  Returns the first planned footstep, used as the immediate swing target.

**FootstepMPCPlanner**

- `__init__(...)`  
  Initializes step timing, horizon length, step width, and desired velocity.

- `reset()`  
  Resets internal phase state and cached plan.

- `update_phase(t)`  
  Advances swing/stance phase based on elapsed time.

- `phase(t)`  
  Returns normalized swing phase \([0, 1]\).

- `current_swing()`  
  Returns which foot is currently swinging.

- `build_plan(trunk_state, omega, left_pos, right_pos)`  
  Computes a capture-point–inspired sequence of foot placements over the MPC horizon.

**ModelPredictiveController**

- `__init__(...)`  
  Initializes LIPM parameters, footstep planner, swing-leg settings, and nominal posture targets.

- `_roty(a)`  
  Returns a pitch rotation matrix.

- `_world_to_body(...)`, `_body_to_world(...)`  
  Coordinate frame transformations between world and trunk frames.

- `_predict_traj(trunk_state, foot_seq)`  
  Rolls out LIPM dynamics over the horizon to predict trunk position and velocity.

- `_build_foot_sequence(plan, left_pos, right_pos)`  
  Builds the stance-foot sequence used for LIPM prediction.

- `_compute_wrench(...)`  
  Computes the desired trunk wrench using LIPM feedforward dynamics combined with PD stabilization.

- `_allocate_contact(...)`  
  Allocates the desired wrench into contact forces using the shared QP solver and maps them to joint torques.

- `_posture_torque(...)`  
  Computes joint-space PD torques for posture stabilization.

- `_ik_swing_leg_body(...)`  
  Solves a planar two-link inverse kinematics problem for swing-leg placement.

- `__call__(model, data, t)`  
  Main MPC control loop that updates phase, plans footsteps, predicts dynamics, allocates contact forces, generates swing motion, applies posture stabilization, clips torques, and outputs the final command.

#### `visualization_utils_walking.py`

- `save_walking_controller_plots(...)`  
  Generates plots for trunk motion, joint states, torques, and contact forces during walking.

### Plots (Task 2 – Walking)

![Hip joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2_walking/hip_q_qd_tau_vs_time.png)

**Figure 7.** Hip joint angles, velocities, and torques over time for the Task 2 Walking

![Knee joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2_walking/knee_q_qd_tau_vs_time.png)

**Figure 8.** Knee joint angles, velocities, and torques over time for the Task 2 Walking

![Walking trunk state](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2_walking/walking_root_state.png)

**Figure 9.** Trunk position, velocity, and pitch angle over time during walking. The MPC controller stabilizes the trunk while tracking forward motion and maintaining balance.

![Walking joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2_walking/walking_joint_states.png)

**Figure 10.** Joint angles, velocities, and commanded torques over time for the walking task. Joint motions are smooth and actuator torques remain within specified limits throughout locomotion.

![Walking contact forces](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2_walking/walking_contact_forces.png)

**Figure 11.** Ground contact forces for the left and right feet during walking. Contact forces remain within friction and vertical force bounds, demonstrating stable foot–ground interaction.

![Walking torques](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task2_walking/walking_torques.png)

**Figure 12.** Walking Torques

### Simulation video link (Task 2 Walking):
https://drive.google.com/file/d/1srsVA-XzIlJfdCdn-KH3QcLx_JqoCqYo/view?usp=sharing

## 4. Task 3: Extended Walking with MPC, QP Contact Allocation, and Logging

### Approach

Task 3 builds on the MPC walking controller from Task 2 by emphasizing **robust execution, logging, and visualization** during sustained walking.

At each control step:

1. A horizon-based footstep plan is generated using capture-point–inspired logic.
2. Trunk dynamics are predicted over the horizon using a Linear Inverted Pendulum Model (LIPM).
3. A desired trunk wrench is computed from LIPM feedforward dynamics and PD stabilization.
4. The desired wrench is allocated to foot contact forces using the shared QP solver.
5. Contact forces are mapped to joint torques via foot Jacobians.
6. Swing-leg motion is generated using a latched target, parabolic height profile, and planar inverse kinematics.
7. Joint-space posture stabilization is applied and torques are clipped to actuator limits.
8. Trunk state, joint state, contact forces, and footsteps are logged for post-analysis, and optional video output is recorded.

---

### Code Structure

#### `task3.py`

**WalkingProfile**

- Defines walking parameters including desired speed, step timing, horizon length, and simulation duration.

---

**WalkingMPCController**

- `__init__(profile)`  
  Initializes the MPC controller, logging buffers, and optional video recording state.

- `enable_video_recording(model)`  
  Sets up an offscreen renderer for video capture.

- `__call__(model, data, t)`  
  Executes the MPC walking controller, logs trunk and joint states, contact forces, and planned footsteps, and optionally records video frames.

- `save_plots(output_dir)`  
  Generates and saves diagnostic walking plots using shared visualization utilities.

- `save_video(output_dir)`  
  Writes the recorded walking video to disk.

---

**Main Entry**

- Parses command-line arguments.
- Initializes the walking profile and controller.
- Runs the simulation using `run_simulation(...)`.
- Saves plots, additional body position/velocity figures, and optional video output.

### Plots (Task 3)

![Hip joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task3/hip_q_qd_tau_vs_time.png)

**Figure 13.** Hip joint angles, velocities, and torques over time for the Task 3

![Knee joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task3/knee_q_qd_tau_vs_time.png)

**Figure 14.** Knee joint angles, velocities, and torques over time for the Task 3

![Walking trunk state](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task3/walking_root_state.png)

**Figure 15.** Trunk position, velocity, and pitch angle over time during walking. The MPC controller stabilizes the trunk while tracking forward motion and maintaining balance.

![Walking joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task3/walking_joint_states.png)

**Figure 16.** Joint angles, velocities, and commanded torques over time for the Task 3. Joint motions are smooth and actuator torques remain within specified limits throughout locomotion.

![Walking contact forces](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task3/walking_contact_forces.png)

**Figure 17.** Ground contact forces for the left and right feet for Task 3. Contact forces remain within friction and vertical force bounds, demonstrating stable foot–ground interaction.

![Walking Torques](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task3/walking_torques.png)

**Figure 18.** Task 3 Torques

### Simulation video link (Task 3):
https://drive.google.com/file/d/10XK_dX_xzXEQCMMN-UlMxpUfWaAQ0V89/view?usp=sharing

## 5. Task 4: Stair Climbing with Terrain-Aware MPC and QP Contact Allocation

### Approach

Task 4 extends the MPC walking controller to enable **stair climbing** by incorporating **terrain-aware foot placement** and additional safety constraints.

The staircase geometry is predefined in the MuJoCo model using step geoms (`step0`–`step4`) with fixed rise and run dimensions.

At each control step:

1. A horizon-based footstep plan is generated using MPC, with parameters tuned for stair ascent.
2. Trunk dynamics are predicted using a Linear Inverted Pendulum Model (LIPM).
3. A desired trunk wrench is computed from LIPM feedforward dynamics and PD stabilization.
4. The wrench is allocated to feasible foot contact forces using the shared QP solver.
5. Contact forces are mapped to joint torques via foot Jacobians.
6. Swing-leg motion uses increased swing-foot clearance to safely clear stair edges.
7. A safety constraint optionally enforces that **only foot geoms may contact stair geoms**, preventing body or leg collisions.
8. Torques are clipped to actuator limits, and optional offscreen video is recorded.

---

### Code Structure

#### `task4.py`

**Task4StairController**

- `__init__(v_des, step_time, horizon_steps, swing_height)`  
  Initializes the stair-aware MPC controller, increases swing-foot clearance, and sets up optional video recording.

- `enable_video_recording(model)`  
  Initializes an offscreen renderer for stair-climbing video capture.

- `__call__(model, data, t)`  
  Executes the MPC stair-climbing controller, returns joint torques and debugging information, and records video frames if enabled.

- `save_video(output_dir)`  
  Writes the recorded stair-climbing video to disk.

---

**Stair Safety Constraint**

- `_no_nonfoot_step_contacts(...)`  
  Enforces a safety rule that only foot sphere geoms may contact stair geoms.  
  Any body or leg contact with a step is reported as a violation (optional but recommended).

---

#### `mpc_controller_stair.py`

This module implements a **terrain-aware model-predictive trunk controller** for stair climbing.  
Unlike the flat-ground walking controller, it explicitly accounts for **step geometry**, **support height changes**, and **edge avoidance** while preserving the same MPC + QP control structure.

**Terrain-Aware Swing Foot Control**

- Planned footsteps are projected onto stair tread surfaces using ray casting.
- Foot targets are nudged away from stair edges to reduce collision risk.
- Swing trajectories use increased clearance and a parabolic height profile to safely clear stair risers.

---

**Main Entry**

- Parses command-line arguments for simulation time, speed, horizon length, swing clearance, and safety checks.
- Runs the simulation using `run_simulation(...)` with an optional terrain-aware constraint checker.
- Saves stair-climbing video output if enabled.

### Plots (Task 4)

![Task 4 body height tracking](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task4/body_height_vs_time.png)

**Figure 19.** Trunk vertical position over time during stair climbing (Task 4).  
The controller maintains stable height regulation while ascending discrete step elevations.

![Task 4 hip joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task4/hip_q_qd_tau_vs_time.png)

**Figure 20.** Hip joint angles, velocities, and commanded torques during stair climbing.  
Hip motion adapts to increased elevation while remaining within joint and actuator limits.

![Task 4 knee joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task4/knee_q_qd_tau_vs_time.png)

**Figure 21.** Knee joint angles, velocities, and torques over time for Task 4 stair ascent.  
Increased knee flexion enables foot clearance over stair edges while satisfying physical constraints.

### Simulation video link (Task 4):
https://drive.google.com/file/d/1U7_SGMvqCjaq9kQYiT1ouYH6RBCOEjUn/view?usp=sharing