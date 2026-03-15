"""
Kalman Filter Milestone-2 — Plotting & Analysis
Generates all required plots:
  1. Time-series: position, velocity, acceleration, jerk (x,y,z) for one joint
  2. True vs Noisy vs LKF vs EKF position comparison
  3. LKF vs EKF comparison with RMSE table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings("ignore")

# ─── paths ───────────────────────────────────────────────────────────────────
TRUE_CSV  = "../data/true.csv"
NOISY_CSV = "../data/noisy.csv"
LKF_CSV   = "../output/lkf_output.csv"
EKF_CSV   = "../output/ekf_output.csv"
OUT_DIR   = "../plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── load data ────────────────────────────────────────────────────────────────
print("Loading CSVs...")
true_df  = pd.read_csv(TRUE_CSV)
noisy_df = pd.read_csv(NOISY_CSV)
lkf_df   = pd.read_csv(LKF_CSV)
ekf_df   = pd.read_csv(EKF_CSV)
T = len(true_df)
time = np.arange(T) / 100.0  # 100 Hz => seconds

print(f"  Frames: {T},  Duration: {time[-1]:.2f} s")

# ─── joint selection: pelvis (joint 0) ───────────────────────────────────────
JOINT = "pelvis"
AXES  = ["x", "y", "z"]

# True and noisy positions (only x,y,z available in raw CSVs)
true_pos  = {ax: true_df[f"{JOINT}_{ax}"].values  for ax in AXES}
noisy_pos = {ax: noisy_df[f"{JOINT}_{ax}"].values for ax in AXES}

# LKF columns: pelvis_px, pelvis_vx, pelvis_ax, pelvis_jx, pelvis_py, ...
lkf_states = {}
ekf_states = {}
state_map = {"px":0,"vx":1,"ax":2,"jx":3,"py":4,"vy":5,"ay":6,"jy":7,"pz":8,"vz":9,"az":10,"jz":11}

for key in state_map:
    col = f"{JOINT}_{key}"
    lkf_states[key] = lkf_df[col].values if col in lkf_df.columns else np.zeros(T)
    ekf_states[key] = ekf_df[col].values if col in ekf_df.columns else np.zeros(T)

# ─── helpers ─────────────────────────────────────────────────────────────────
COLOR_TRUE  = "#1a1a2e"
COLOR_NOISY = "#e94560"
COLOR_LKF   = "#0f3460"
COLOR_EKF   = "#16213e"
COLOR_EKF2  = "#e94560"

STYLE = dict(linewidth=0.9)

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Position time-series (True vs Noisy vs LKF vs EKF)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlot 1: Position comparison...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Position — Joint: {JOINT.capitalize()}\nTrue vs Noisy vs LKF vs EKF", fontsize=13)

pos_keys = ["px", "py", "pz"]
ax_labels = ["X (m)", "Y (m)", "Z (m)"]
for i, (pk, lab) in enumerate(zip(pos_keys, ax_labels)):
    ax = axes[i]
    raw_ax = AXES[i]
    ax.plot(time, true_pos[raw_ax],  color=COLOR_TRUE,  label="True",  **STYLE, alpha=0.9)
    ax.plot(time, noisy_pos[raw_ax], color=COLOR_NOISY, label="Noisy", **STYLE, alpha=0.5, linewidth=0.5)
    ax.plot(time, lkf_states[pk],    color=COLOR_LKF,   label="LKF",   **STYLE, linestyle="--")
    ax.plot(time, ekf_states[pk],    color=COLOR_EKF2,  label="EKF",   **STYLE, linestyle="-.")
    ax.set_ylabel(lab, fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8, ncol=4)

axes[-1].set_xlabel("Time (s)", fontsize=9)
plt.tight_layout()
savefig("01_position_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Full state time-series (LKF): pos, vel, acc, jerk
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 2: LKF full state time-series...")
state_groups = [
    ("Position (m)",       ["px","py","pz"]),
    ("Velocity (m/s)",     ["vx","vy","vz"]),
    ("Acceleration (m/s²)",["ax","ay","az"]),
    ("Jerk (m/s³)",        ["jx","jy","jz"]),
]
colors = ["#e63946","#2a9d8f","#264653"]

fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
fig.suptitle(f"LKF Full State — Joint: {JOINT.capitalize()}", fontsize=13)

for row, (ylabel, keys) in enumerate(state_groups):
    for col, (key, color) in enumerate(zip(keys, colors)):
        ax = axes[row][col]
        ax.plot(time, lkf_states[key], color=color, linewidth=0.8)
        ax.set_ylabel(ylabel if col == 0 else "", fontsize=7)
        ax.set_title(key.upper(), fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 3:
            ax.set_xlabel("Time (s)", fontsize=8)

plt.tight_layout()
savefig("02_lkf_full_state.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Full state time-series (EKF)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 3: EKF full state time-series...")
fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
fig.suptitle(f"EKF Full State — Joint: {JOINT.capitalize()}", fontsize=13)

for row, (ylabel, keys) in enumerate(state_groups):
    for col, (key, color) in enumerate(zip(keys, colors)):
        ax = axes[row][col]
        ax.plot(time, ekf_states[key], color=color, linewidth=0.8)
        ax.set_ylabel(ylabel if col == 0 else "", fontsize=7)
        ax.set_title(key.upper(), fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 3:
            ax.set_xlabel("Time (s)", fontsize=8)

plt.tight_layout()
savefig("03_ekf_full_state.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — LKF vs EKF side-by-side position
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 4: LKF vs EKF comparison...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"LKF vs EKF Position Comparison — Joint: {JOINT.capitalize()}", fontsize=13)

for i, pk in enumerate(["px","py","pz"]):
    ax = axes[i]
    ax.plot(time, true_pos[AXES[i]],  color=COLOR_TRUE, label="True",  **STYLE, alpha=0.9, linewidth=1.2)
    ax.plot(time, lkf_states[pk],     color=COLOR_LKF,  label="LKF",   **STYLE, linestyle="--")
    ax.plot(time, ekf_states[pk],     color=COLOR_EKF2, label="EKF",   **STYLE, linestyle="-.")
    ax.set_ylabel(f"p{AXES[i]} (m)", fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=9, ncol=3)

axes[-1].set_xlabel("Time (s)", fontsize=9)
plt.tight_layout()
savefig("04_lkf_vs_ekf_position.png")

# ═══════════════════════════════════════════════════════════════════════════════
# RMSE computation
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing RMSE...")
rmse_data = []
for jt_name in true_df.columns[::3]:  # iterate joint names
    jt = jt_name.replace("_x","")
    for ax, pk, py_, pz_ in [("x","px","py","pz")]:
        pass

# Compute for all joints
all_lkf_rmse_pos = []
all_ekf_rmse_pos = []

for jt_name in [c for c in true_df.columns if c.endswith("_x")]:
    jt = jt_name.replace("_x","")
    for suffix, lk, ek in [("x","px","px"),("y","py","py"),("z","pz","pz")]:
        col_true = f"{jt}_{suffix}"
        col_lkf  = f"{jt}_p{suffix}"
        col_ekf  = f"{jt}_p{suffix}"
        if col_true in true_df.columns and col_lkf in lkf_df.columns:
            t_vals = true_df[col_true].values
            l_vals = lkf_df[col_lkf].values
            e_vals = ekf_df[col_ekf].values
            all_lkf_rmse_pos.append(np.sqrt(np.mean((t_vals - l_vals)**2)))
            all_ekf_rmse_pos.append(np.sqrt(np.mean((t_vals - e_vals)**2)))

mean_lkf = np.mean(all_lkf_rmse_pos) if all_lkf_rmse_pos else float('nan')
mean_ekf = np.mean(all_ekf_rmse_pos) if all_ekf_rmse_pos else float('nan')

# Per-axis RMSE for chosen joint
rmse_table = {}
for i, (pk, ax_raw) in enumerate(zip(["px","py","pz"], AXES)):
    t_v = true_pos[ax_raw]
    l_v = lkf_states[pk]
    e_v = ekf_states[pk]
    rmse_table[ax_raw] = {
        "LKF": np.sqrt(np.mean((t_v - l_v)**2)),
        "EKF": np.sqrt(np.mean((t_v - e_v)**2)),
    }

print(f"\n  RMSE for joint '{JOINT}':")
print(f"  {'Axis':<8} {'LKF':>10} {'EKF':>10}")
print(f"  {'-'*30}")
for ax_raw in AXES:
    print(f"  {ax_raw:<8} {rmse_table[ax_raw]['LKF']:>10.5f} {rmse_table[ax_raw]['EKF']:>10.5f}")
print(f"\n  Mean position RMSE — LKF: {mean_lkf:.5f} m,  EKF: {mean_ekf:.5f} m")

# Save RMSE to text file
with open(os.path.join(OUT_DIR, "rmse_results.txt"), "w") as f:
    f.write(f"RMSE Results — Joint: {JOINT}\n")
    f.write(f"{'Axis':<8} {'LKF':>12} {'EKF':>12}\n")
    f.write("-"*35 + "\n")
    for ax_raw in AXES:
        f.write(f"{ax_raw:<8} {rmse_table[ax_raw]['LKF']:>12.6f} {rmse_table[ax_raw]['EKF']:>12.6f}\n")
    f.write(f"\nMean LKF RMSE: {mean_lkf:.6f}\n")
    f.write(f"Mean EKF RMSE: {mean_ekf:.6f}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — RMSE bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4))
x_pos = np.arange(3)
w = 0.35
lkf_vals = [rmse_table[ax_]['LKF'] for ax_ in AXES]
ekf_vals  = [rmse_table[ax_]['EKF'] for ax_ in AXES]
ax.bar(x_pos - w/2, lkf_vals, w, label="LKF", color=COLOR_LKF, alpha=0.85)
ax.bar(x_pos + w/2, ekf_vals,  w, label="EKF", color=COLOR_EKF2, alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(["X", "Y", "Z"])
ax.set_ylabel("RMSE (m)")
ax.set_title(f"Position RMSE — {JOINT.capitalize()}")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
savefig("05_rmse_comparison.png")

print("\nAll plots saved to ../plots/")
