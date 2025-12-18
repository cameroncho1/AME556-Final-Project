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

**Figure 1.** Hip joint angles, velocities, and torques over time for the Task 1 PD
controller. All values remain within specified physical limits.

![Hip joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task1/hip_q_qd_tau_vs_time.png)

![Knee joint states](https://raw.githubusercontent.com/cameroncho1/AME556-Final-Project/main/plots/task1/knee_q_qd_tau_vs_time.png)


**Figure 2.** Knee joint angles, velocities, and torques over time for the Task 1 PD
controller. Actuator torques remain within saturation limits and joint constraints
are respected throughout the simulation.


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

- Trunk height tracking  
- Joint angles and velocities  
- Joint torques vs. limits  
- Contact forces vs. friction and force bounds  

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

- Trunk position and velocity  
- Joint angles and velocities  
- Joint torques  
- Contact forces during walking  