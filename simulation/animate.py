"""
Kalman Filter Milestone-2 — 3D Full-Body Walking Simulation
Animates 3 side-by-side skeleton views: Measured, LKF, EKF
Exports to animation.mp4 (or animation.gif if ffmpeg unavailable)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os, warnings
warnings.filterwarnings("ignore")

# ─── paths ───────────────────────────────────────────────────────────────────
NOISY_CSV = "../data/noisy.csv"
LKF_CSV   = "../output/lkf_output.csv"
EKF_CSV   = "../output/ekf_output.csv"
OUT_DIR   = "../simulation"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── joint list (must match CSV column order) ─────────────────────────────────
JOINT_NAMES = [
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
]
N_JOINTS = len(JOINT_NAMES)
jidx = {name: i for i, name in enumerate(JOINT_NAMES)}

# ─── skeleton connectivity (pairs of joint indices) ───────────────────────────
BONES = [
    # Spine
    ("pelvis","L5"), ("L5","L3"), ("L3","T12"), ("T12","T8"), ("T8","neck"), ("neck","head"),
    # Right arm
    ("T8","shoulderRight"), ("shoulderRight","upperArmRight"),
    ("upperArmRight","forearmRight"), ("forearmRight","handRight"),
    # Left arm
    ("T8","shoulderLeft"), ("shoulderLeft","upperArmLeft"),
    ("upperArmLeft","forearmLeft"), ("forearmLeft","handLeft"),
    # Right leg
    ("pelvis","upperLegRight"), ("upperLegRight","lowerLegRight"),
    ("lowerLegRight","footRight"), ("footRight","toeRight"),
    # Left leg
    ("pelvis","upperLegLeft"), ("upperLegLeft","lowerLegLeft"),
    ("lowerLegLeft","footLeft"), ("footLeft","toeLeft"),
]
BONE_PAIRS = [(jidx[a], jidx[b]) for a, b in BONES]

# ─── load data ────────────────────────────────────────────────────────────────
print("Loading data...")
noisy_df = pd.read_csv(NOISY_CSV)
lkf_df   = pd.read_csv(LKF_CSV)
ekf_df   = pd.read_csv(EKF_CSV)

T = min(len(noisy_df), len(lkf_df), len(ekf_df))
# Subsample to max 500 frames for animation speed
STEP = max(1, T // 500)
frames_idx = list(range(0, T, STEP))
N_FRAMES = len(frames_idx)
print(f"  Total frames: {T}, animating {N_FRAMES} (step={STEP})")

def extract_positions(df, frame_indices):
    """Returns array of shape (N_FRAMES, N_JOINTS, 3)"""
    pos = np.zeros((len(frame_indices), N_JOINTS, 3))
    for i, fi in enumerate(frame_indices):
        for j, jname in enumerate(JOINT_NAMES):
            # Noisy CSV has jname_x, jname_y, jname_z
            # LKF/EKF have jname_px, jname_py, jname_pz
            for ax_idx, suffix in enumerate(["_x","_y","_z"]):
                col = f"{jname}{suffix}"
                p_col = f"{jname}_p{suffix[1]}"  # e.g. pelvis_px
                if col in df.columns:
                    pos[i, j, ax_idx] = df[col].iloc[fi]
                elif p_col in df.columns:
                    pos[i, j, ax_idx] = df[p_col].iloc[fi]
    return pos

print("Extracting positions...")
pos_noisy = extract_positions(noisy_df, frames_idx)
pos_lkf   = extract_positions(lkf_df,   frames_idx)
pos_ekf   = extract_positions(ekf_df,   frames_idx)

# ─── axis limits (common across all 3 panels) ─────────────────────────────────
all_pos = np.concatenate([pos_noisy, pos_lkf, pos_ekf], axis=0)
pad = 0.3
xlim = (all_pos[:,:,0].min()-pad, all_pos[:,:,0].max()+pad)
ylim = (all_pos[:,:,1].min()-pad, all_pos[:,:,1].max()+pad)
zlim = (all_pos[:,:,2].min()-pad, all_pos[:,:,2].max()+pad)

# ─── set up figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
fig.patch.set_facecolor("#0d1117")

axes3d = []
for panel in range(3):
    ax = fig.add_subplot(1, 3, panel+1, projection='3d')
    ax.set_facecolor("#0d1117")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X", fontsize=7, color="gray")
    ax.set_ylabel("Y", fontsize=7, color="gray")
    ax.set_zlabel("Z", fontsize=7, color="gray")
    ax.tick_params(colors="gray", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    axes3d.append(ax)

titles  = ["Measured (Noisy)", "LKF Estimate", "EKF Estimate"]
colors  = ["#e94560", "#4cc9f0", "#06d6a0"]

for ax, title, color in zip(axes3d, titles, colors):
    ax.set_title(title, color=color, fontsize=9, pad=3)

# ─── draw one frame ───────────────────────────────────────────────────────────
def draw_skeleton(ax, pos_frame, color):
    ax.cla()
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="gray", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.15)

    # Draw bones
    for a_idx, b_idx in BONE_PAIRS:
        xs = [pos_frame[a_idx, 0], pos_frame[b_idx, 0]]
        ys = [pos_frame[a_idx, 1], pos_frame[b_idx, 1]]
        zs = [pos_frame[a_idx, 2], pos_frame[b_idx, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=0.85)

    # Draw joints
    ax.scatter(pos_frame[:, 0], pos_frame[:, 1], pos_frame[:, 2],
               c=color, s=12, alpha=0.9, depthshade=False)

datasets = [pos_noisy, pos_lkf, pos_ekf]

def animate(frame_num):
    for ax, pos_data, color, title in zip(axes3d, datasets, colors, titles):
        draw_skeleton(ax, pos_data[frame_num], color)
        ax.set_title(title, color=color, fontsize=9, pad=3)
    fig.suptitle(f"3D Full-Body Gait — Frame {frames_idx[frame_num]}",
                 color="white", fontsize=11, y=1.01)
    return []

print("Building animation...")
ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES,
                               interval=50, blit=False)

# Try mp4 first, fall back to gif
try:
    Writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    out_path = os.path.join(OUT_DIR, "animation.mp4")
    ani.save(out_path, writer=Writer, dpi=100,
             savefig_kwargs={"facecolor": "#0d1117"})
    print(f"Saved: {out_path}")
except Exception as e:
    print(f"FFMpeg not available ({e}), saving as GIF...")
    out_path = os.path.join(OUT_DIR, "animation.gif")
    ani.save(out_path, writer="pillow", fps=20,
             savefig_kwargs={"facecolor": "#0d1117"})
    print(f"Saved: {out_path}")

plt.close()
print("Simulation done.")
