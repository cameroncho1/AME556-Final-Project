import os
import numpy as np
import matplotlib.pyplot as plt
# import visualization_utils
HERE = os.path.dirname(os.path.abspath(__file__))

def save_walking_controller_plots(log_data: dict, output_dir: str = None):
    """
    Save diagnostic plots for walking controller.
    Args:
        log_data: Dictionary containing logged simulation data with keys:
            - time: list of timestamps
            - root_pos: list of [x, z, theta] arrays
            - root_vel: list of [xd, zd, thetad] arrays
            - joint_pos: list of joint position arrays
            - joint_vel: list of joint velocity arrays
            - tau_cmd: list of commanded torque arrays
            - contact_forces: list of [Fx_r, Fy_r, Fx_l, Fy_l] arrays
            - footsteps: list of planned footsteps (arrays)
        output_dir: Directory to save plots (default: task2_walking_result/plots)
    """
    if not log_data.get("time"):
        print("[INFO] No logged data to plot.")
        return
    out_dir = output_dir or os.path.join(HERE, "task2_walking_result", "plots")
    os.makedirs(out_dir, exist_ok=True)
    time_arr = np.asarray(log_data["time"])
    root_pos = np.vstack(log_data["root_pos"])
    root_vel = np.vstack(log_data["root_vel"])
    joint_pos = np.vstack(log_data["joint_pos"])
    joint_vel = np.vstack(log_data["joint_vel"])
    tau_cmd = np.vstack(log_data["tau_cmd"])
    contact_forces = np.vstack(log_data["contact_forces"])
    # Plot root position and velocity
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs1[0].plot(time_arr, root_pos[:, 0], label="x (root)")
    axs1[0].plot(time_arr, root_pos[:, 1], label="z (root)")
    axs1[0].set_ylabel("Root pos [m]")
    axs1[0].grid(True)
    axs1[0].legend(loc="upper right")
    axs1[1].plot(time_arr, root_vel[:, 0], label="xd")
    axs1[1].plot(time_arr, root_vel[:, 1], label="zd")
    axs1[1].set_ylabel("Root vel [m/s]")
    axs1[1].set_xlabel("Time [s]")
    axs1[1].grid(True)
    axs1[1].legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "walking_root_state.png"), dpi=200)
    plt.close(fig1)
    # Plot joint angles and velocities
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for idx in range(4):
        axs2[0].plot(time_arr, joint_pos[:, idx], label=f"q{idx+1}")
    axs2[0].set_ylabel("Joint angle [rad]")
    axs2[0].grid(True)
    axs2[0].legend(loc="upper right")
    for idx in range(4):
        axs2[1].plot(time_arr, joint_vel[:, idx], label=f"q{idx+1}_dot")
    axs2[1].set_ylabel("Joint vel [rad/s]")
    axs2[1].set_xlabel("Time [s]")
    axs2[1].grid(True)
    axs2[1].legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "walking_joint_states.png"), dpi=200)
    plt.close(fig2)
    # Plot torques
    fig3, axs3 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for idx in range(4):
        axs3[idx].plot(time_arr, tau_cmd[:, idx], label="applied", linewidth=1.8)
        axs3[idx].set_ylabel(f"tau{idx+1} [Nm]")
        axs3[idx].grid(True)
        if idx == 0:
            axs3[idx].legend(loc="upper right")
    axs3[-1].set_xlabel("Time [s]")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "walking_torques.png"), dpi=200)
    plt.close(fig3)
    # Plot contact forces
    fig4, axs4 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs4[0].plot(time_arr, contact_forces[:, 0], label="Fx_r")
    axs4[0].plot(time_arr, contact_forces[:, 2], label="Fx_l")
    axs4[0].set_ylabel("Fx [N]")
    axs4[0].grid(True)
    axs4[0].legend(loc="upper right")
    axs4[1].plot(time_arr, contact_forces[:, 1], label="Fz_r")
    axs4[1].plot(time_arr, contact_forces[:, 3], label="Fz_l")
    axs4[1].set_ylabel("Fz [N]")
    axs4[1].grid(True)
    axs4[1].legend(loc="upper right")
    axs4[1].set_xlabel("Time [s]")
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "walking_contact_forces.png"), dpi=200)
    plt.close(fig4)
    print(f"[INFO] Saved Task 2 walking diagnostic plots to: {out_dir}")

