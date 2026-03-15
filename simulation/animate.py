"""
animate.py  
Produces a 3-panel side-by-side 3D skeleton animation:
  Panel 1 — LKF skeleton
  Panel 2 — EKF skeleton
  Panel 3 — Overlay (LKF blue, EKF red)
Exports as  ../output/skeleton_animation.mp4
Requires:  ffmpeg  (apt-get install ffmpeg)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import os, re

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "..", "output")
OUTPUT = os.path.join(BASE, "..", "output")
os.makedirs(OUTPUT, exist_ok=True)

lkf = pd.read_csv(os.path.join(DATA, "lkf_output.csv"))
ekf = pd.read_csv(os.path.join(DATA, "ekf_output.csv"))

JOINTS = list(dict.fromkeys([
    re.sub(r'_(px|vx|ax|jx|py|vy|ay|jy|pz|vz|az|jz)$', '', c)
    for c in lkf.columns
]))
N_JOINTS = len(JOINTS)   # 23

# ── skeleton connectivity (parent → child) ─────────────────────────────────────
# Joint index map
JI = {j: i for i, j in enumerate(JOINTS)}

BONES = [
    # spine
    ("pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"),
    ("T8", "neck"), ("neck", "head"),
    # right arm
    ("T8", "shoulderRight"), ("shoulderRight", "upperArmRight"),
    ("upperArmRight", "forearmRight"), ("forearmRight", "handRight"),
    # left arm
    ("T8", "shoulderLeft"), ("shoulderLeft", "upperArmLeft"),
    ("upperArmLeft", "forearmLeft"), ("forearmLeft", "handLeft"),
    # right leg
    ("pelvis", "upperLegRight"), ("upperLegRight", "lowerLegRight"),
    ("lowerLegRight", "footRight"), ("footRight", "toeRight"),
    # left leg
    ("pelvis", "upperLegLeft"), ("upperLegLeft", "lowerLegLeft"),
    ("lowerLegLeft", "footLeft"), ("footLeft", "toeLeft"),
]

def get_positions(df, frame):
    """Return (23, 3) array of joint positions at given frame."""
    row = df.iloc[frame]
    pos = np.zeros((N_JOINTS, 3))
    for i, jt in enumerate(JOINTS):
        pos[i, 0] = row[f"{jt}_px"]
        pos[i, 1] = row[f"{jt}_py"]
        pos[i, 2] = row[f"{jt}_pz"]
    return pos

# ── compute global axis limits ────────────────────────────────────────────────
SUBSAMPLE = 10   # compute limits from every 10th frame
all_vals = []
for f in range(0, len(lkf), SUBSAMPLE):
    p = get_positions(lkf, f)
    all_vals.append(p)
all_vals = np.vstack(all_vals)
margin = 0.3
xlim = (all_vals[:,0].min() - margin, all_vals[:,0].max() + margin)
ylim = (all_vals[:,1].min() - margin, all_vals[:,1].max() + margin)
zlim = (all_vals[:,2].min() - margin, all_vals[:,2].max() + margin)

C_LKF = "#2563EB"
C_EKF = "#DC2626"

def draw_skeleton(ax, pos, color, alpha=1.0, lw=1.8):
    """Draw skeleton lines on a 3D axis."""
    for pa, ch in BONES:
        if pa in JI and ch in JI:
            i, j = JI[pa], JI[ch]
            ax.plot([pos[i,0], pos[j,0]],
                    [pos[i,1], pos[j,1]],
                    [pos[i,2], pos[j,2]],
                    color=color, alpha=alpha, lw=lw)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=color, s=8, alpha=alpha)

def setup_ax(ax, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X", fontsize=7, labelpad=1)
    ax.set_ylabel("Y", fontsize=7, labelpad=1)
    ax.set_zlabel("Z", fontsize=7, labelpad=1)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=15, azim=45)

# ── animation ─────────────────────────────────────────────────────────────────
TOTAL_FRAMES = len(lkf)
STEP = 3            # animate every 3rd frame  (3040 → ~1013 frames at 30fps ≈ 34 s)
FRAMES = list(range(0, TOTAL_FRAMES, STEP))
FPS = 30

fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor("#111827")
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
for ax in [ax1, ax2, ax3]:
    ax.set_facecolor("#1F2937")

time_text = fig.text(0.5, 0.97, '', ha='center', va='top',
                     color='white', fontsize=10)

def init():
    return []

def animate(fi):
    frame = FRAMES[fi]
    t = frame * 0.01

    for ax in [ax1, ax2, ax3]:
        ax.cla()
        setup_ax(ax, "")

    ax1.set_title("LKF Skeleton", fontsize=9, color="white", pad=4)
    ax2.set_title("EKF Skeleton", fontsize=9, color="white", pad=4)
    ax3.set_title("LKF vs EKF Overlay", fontsize=9, color="white", pad=4)

    lkf_pos = get_positions(lkf, frame)
    ekf_pos = get_positions(ekf, frame)

    draw_skeleton(ax1, lkf_pos, C_LKF)
    draw_skeleton(ax2, ekf_pos, C_EKF)
    draw_skeleton(ax3, lkf_pos, C_LKF, alpha=0.8)
    draw_skeleton(ax3, ekf_pos, C_EKF, alpha=0.6, lw=1.2)

    time_text.set_text(f"t = {t:.2f} s  |  frame {frame}/{TOTAL_FRAMES}")
    return []

ani = animation.FuncAnimation(
    fig, animate, frames=len(FRAMES),
    init_func=init, interval=1000/FPS, blit=False
)

out_mp4 = os.path.join(OUTPUT, "skeleton_animation.mp4")
writer = animation.FFMpegWriter(fps=FPS, bitrate=1800,
                                extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
print(f"Rendering {len(FRAMES)} frames → {out_mp4} ...")
ani.save(out_mp4, writer=writer, dpi=100)
plt.close()
print(f"  ✓ Saved: {out_mp4}")
