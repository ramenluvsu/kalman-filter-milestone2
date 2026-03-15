"""
generate_plots.py  
Produces 5 plots from lkf_output.csv and ekf_output.csv:
  1. position_comparison.png       — raw position traces for a representative joint
  2. lkf_full_state.png            — 12-subplot grid: all state variables for LKF (pelvis)
  3. ekf_full_state.png            — 12-subplot grid: all state variables for EKF (pelvis)
  4. lkf_vs_ekf_overlay.png        — overlay of LKF vs EKF position for 3 joints
  5. rmse_bar_chart.png            — per-joint RMSE bar chart
All plots are saved to  ../plots/
RMSE numbers are also written to  ../output/rmse_results.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, re

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "..", "output")
PLOTS  = os.path.join(BASE, "..", "plots")
OUTPUT = os.path.join(BASE, "..", "output")
os.makedirs(PLOTS,  exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
lkf = pd.read_csv(os.path.join(DATA, "lkf_output.csv"))
ekf = pd.read_csv(os.path.join(DATA, "ekf_output.csv"))

# Joint list (23 joints, order as in CSV)
JOINTS = list(dict.fromkeys([
    re.sub(r'_(px|vx|ax|jx|py|vy|ay|jy|pz|vz|az|jz)$', '', c)
    for c in lkf.columns
]))

n_steps = len(lkf)
dt = 0.01          # 100 Hz
time = np.arange(n_steps) * dt   # seconds

# ── colour palette ─────────────────────────────────────────────────────────────
C_LKF = "#2563EB"   # blue
C_EKF = "#DC2626"   # red
C_RAW = "#6B7280"   # grey

# ── helper: axis label ─────────────────────────────────────────────────────────
STATE_LABELS = {
    "px": "pos x (m)", "py": "pos y (m)", "pz": "pos z (m)",
    "vx": "vel x (m/s)", "vy": "vel y (m/s)", "vz": "vel z (m/s)",
    "ax": "acc x (m/s²)", "ay": "acc y (m/s²)", "az": "acc z (m/s²)",
    "jx": "jerk x", "jy": "jerk y", "jz": "jerk z",
}

# ==============================================================================
# Plot 1 — position comparison (3 joints, x-axis position)
# ==============================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Position Comparison — LKF vs EKF (Selected Joints)", fontsize=14, fontweight="bold")

showcase_joints = ["pelvis", "handRight", "footLeft"]
for ax, jt in zip(axes, showcase_joints):
    col_px = f"{jt}_px"
    ax.plot(time, lkf[col_px], color=C_LKF, lw=1.4, label="LKF", alpha=0.9)
    ax.plot(time, ekf[col_px], color=C_EKF, lw=1.4, label="EKF", alpha=0.9, linestyle="--")
    ax.set_ylabel(f"{jt}\npos x (m)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.set_facecolor("#F9FAFB")

axes[-1].set_xlabel("Time (s)", fontsize=10)
plt.tight_layout()
out = os.path.join(PLOTS, "position_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {out}")

# ==============================================================================
# Plot 2 — LKF full state (12 subplots: 3 axes × 4 state vars, for pelvis)
# ==============================================================================
STATE_VARS = ["p", "v", "a", "j"]
AXES_3D    = ["x", "y", "z"]
LABELS_3D  = ["X", "Y", "Z"]
JOINT_FULL = "pelvis"

fig = plt.figure(figsize=(15, 10))
fig.suptitle(f"LKF Full State — Joint: {JOINT_FULL}", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

ylabels = ["Position (m)", "Velocity (m/s)", "Accel. (m/s²)", "Jerk (m/s³)"]
for row, (sv, yl) in enumerate(zip(STATE_VARS, ylabels)):
    for col, ax3 in enumerate(AXES_3D):
        col_name = f"{JOINT_FULL}_{sv}{ax3}"
        ax = fig.add_subplot(gs[row, col])
        ax.plot(time, lkf[col_name], color=C_LKF, lw=1.2)
        ax.set_title(f"{yl.split()[0]} {LABELS_3D[col]}", fontsize=8, pad=3)
        if col == 0:
            ax.set_ylabel(yl, fontsize=7)
        if row == 3:
            ax.set_xlabel("Time (s)", fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#F0F4FF")

out = os.path.join(PLOTS, "lkf_full_state.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {out}")

# ==============================================================================
# Plot 3 — EKF full state (same structure)
# ==============================================================================
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f"EKF Full State — Joint: {JOINT_FULL}", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

for row, (sv, yl) in enumerate(zip(STATE_VARS, ylabels)):
    for col, ax3 in enumerate(AXES_3D):
        col_name = f"{JOINT_FULL}_{sv}{ax3}"
        ax = fig.add_subplot(gs[row, col])
        ax.plot(time, ekf[col_name], color=C_EKF, lw=1.2)
        ax.set_title(f"{yl.split()[0]} {LABELS_3D[col]}", fontsize=8, pad=3)
        if col == 0:
            ax.set_ylabel(yl, fontsize=7)
        if row == 3:
            ax.set_xlabel("Time (s)", fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#FFF0F0")

out = os.path.join(PLOTS, "ekf_full_state.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {out}")

# ==============================================================================
# Plot 4 — LKF vs EKF overlay for 6 joints, px only
# ==============================================================================
overlay_joints = ["pelvis", "T8", "neck", "upperArmRight", "lowerLegLeft", "handRight"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
fig.suptitle("LKF vs EKF Overlay — Position X (6 Joints)", fontsize=14, fontweight="bold")
axes = axes.flatten()

for ax, jt in zip(axes, overlay_joints):
    col = f"{jt}_px"
    ax.plot(time, lkf[col], color=C_LKF, lw=1.4, label="LKF")
    ax.plot(time, ekf[col], color=C_EKF, lw=1.2, label="EKF", linestyle="--", alpha=0.85)
    ax.set_title(jt, fontsize=10)
    ax.set_ylabel("pos x (m)", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, lw=0.35, alpha=0.5)
    ax.set_facecolor("#F9FAFB")

plt.tight_layout()
out = os.path.join(PLOTS, "lkf_vs_ekf_overlay.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {out}")

# ==============================================================================
# Plot 5 — RMSE bar chart  (LKF vs EKF per joint, position magnitude)
# ==============================================================================
rmse_lkf = []
rmse_ekf = []
for jt in JOINTS:
    lkf_pos = lkf[[f"{jt}_px", f"{jt}_py", f"{jt}_pz"]].values
    ekf_pos = ekf[[f"{jt}_px", f"{jt}_py", f"{jt}_pz"]].values
    # RMSE between LKF and EKF (treating LKF as reference — smoother / lower-noise)
    diff = lkf_pos - ekf_pos
    rmse_lkf.append(float(np.sqrt(np.mean(lkf_pos**2))))   # RMS magnitude
    rmse_ekf.append(float(np.sqrt(np.mean(ekf_pos**2))))
    # Also store inter-filter RMSE
inter_rmse = [float(np.sqrt(np.mean((lkf[[f"{jt}_px",f"{jt}_py",f"{jt}_pz"]].values -
                                      ekf[[f"{jt}_px",f"{jt}_py",f"{jt}_pz"]].values)**2)))
              for jt in JOINTS]

x = np.arange(len(JOINTS))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 6))
bars1 = ax.bar(x - width/2, rmse_lkf, width, label="LKF RMS pos", color=C_LKF, alpha=0.85)
bars2 = ax.bar(x + width/2, rmse_ekf, width, label="EKF RMS pos", color=C_EKF, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(JOINTS, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("RMS Position Magnitude (m)", fontsize=10)
ax.set_title("LKF vs EKF — RMS Position per Joint", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", lw=0.4, alpha=0.5)
ax.set_facecolor("#F9FAFB")
plt.tight_layout()
out = os.path.join(PLOTS, "rmse_bar_chart.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {out}")

# ==============================================================================
# Write RMSE results text file
# ==============================================================================
rmse_path = os.path.join(OUTPUT, "rmse_results.txt")
with open(rmse_path, "w") as f:
    f.write("RMSE Analysis Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'Joint':<20} {'LKF RMS (m)':>14} {'EKF RMS (m)':>14} {'Inter-filter RMSE':>18}\n")
    f.write("-" * 68 + "\n")
    for jt, lr, er, ir in zip(JOINTS, rmse_lkf, rmse_ekf, inter_rmse):
        f.write(f"{jt:<20} {lr:>14.6f} {er:>14.6f} {ir:>18.6f}\n")
    f.write("-" * 68 + "\n")
    f.write(f"{'MEAN':<20} {np.mean(rmse_lkf):>14.6f} {np.mean(rmse_ekf):>14.6f} {np.mean(inter_rmse):>18.6f}\n")
    f.write(f"{'STD':<20} {np.std(rmse_lkf):>14.6f} {np.std(rmse_ekf):>14.6f} {np.std(inter_rmse):>18.6f}\n")

print(f"  ✓ Saved: {rmse_path}")
print("\nAll plots and RMSE results generated successfully.")
