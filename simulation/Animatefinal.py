"""
animate.py  —  Milestone 3
Produces a 3-panel side-by-side 3D skeleton animation:
  Panel 1 — Measured  (raw noisy data from noisy.csv)
  Panel 2 — LKF       (assembly output from lkf_asm_output.csv)
  Panel 3 — EKF       (assembly output from ekf_asm_output.csv)

The three-panel Measured → LKF → EKF layout lets the viewer directly see:
  1. How much noise the filters suppress vs the raw measurement.
  2. How the two assembly filter outputs compare to each other.

Exports as  output/skeleton_animation.mp4
Requires:   ffmpeg  (sudo apt-get install ffmpeg)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import os, re

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")
os.makedirs(OUTPUT, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
# noisy.csv           — raw Cartesian positions with Gaussian noise σ=0.5 m
# lkf_asm_output.csv  — LKF assembly filter output (positions extracted from state)
# ekf_asm_output.csv  — EKF assembly filter output

noisy_path = os.path.join(DATA,   "noisy.csv")
lkf_path   = os.path.join(OUTPUT, "lkf_asm_output.csv")
ekf_path   = os.path.join(OUTPUT, "ekf_asm_output.csv")

for p in [noisy_path, lkf_path, ekf_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

noisy_raw = pd.read_csv(noisy_path)   # columns: pelvis_px, pelvis_py, pelvis_pz, L5_px, ...
lkf       = pd.read_csv(lkf_path)    # columns: pelvis_px, pelvis_vx, ..., toeLeft_jz
ekf       = pd.read_csv(ekf_path)

# ── joint list — derived from LKF output column names ─────────────────────────
JOINTS = list(dict.fromkeys([
    re.sub(r'_(px|vx|ax|jx|py|vy|ay|jy|pz|vz|az|jz)$', '', c)
    for c in lkf.columns
]))
N_JOINTS = len(JOINTS)   # 23
JI = {j: i for i, j in enumerate(JOINTS)}

# ── skeleton connectivity ──────────────────────────────────────────────────────
BONES = [
    ("pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"),
    ("T8", "neck"),   ("neck", "head"),
    ("T8", "shoulderRight"),  ("shoulderRight", "upperArmRight"),
    ("upperArmRight", "forearmRight"), ("forearmRight", "handRight"),
    ("T8", "shoulderLeft"),   ("shoulderLeft", "upperArmLeft"),
    ("upperArmLeft", "forearmLeft"),   ("forearmLeft", "handLeft"),
    ("pelvis", "upperLegRight"), ("upperLegRight", "lowerLegRight"),
    ("lowerLegRight", "footRight"),    ("footRight", "toeRight"),
    ("pelvis", "upperLegLeft"),  ("upperLegLeft", "lowerLegLeft"),
    ("lowerLegLeft", "footLeft"),      ("footLeft", "toeLeft"),
]

# ── position extraction helpers ────────────────────────────────────────────────
def get_positions_filtered(df, frame):
    """
    Extract (N_JOINTS, 3) position array from a filter output CSV.
    Only the _px, _py, _pz columns are read; velocity/accel/jerk ignored.
    """
    row = df.iloc[frame]
    pos = np.zeros((N_JOINTS, 3))
    for i, jt in enumerate(JOINTS):
        pos[i, 0] = row[f"{jt}_px"]
        pos[i, 1] = row[f"{jt}_py"]
        pos[i, 2] = row[f"{jt}_pz"]
    return pos

def get_positions_noisy(df, frame):
    """
    Extract (N_JOINTS, 3) position array from noisy.csv.
    noisy.csv uses raw position column names: pelvis_x, pelvis_y, pelvis_z
    (no 'p' prefix — distinct from the filter output _px/_py/_pz format).
    """
    row = df.iloc[frame]
    pos = np.zeros((N_JOINTS, 3))
    for i, jt in enumerate(JOINTS):
        pos[i, 0] = row[f"{jt}_x"]
        pos[i, 1] = row[f"{jt}_y"]
        pos[i, 2] = row[f"{jt}_z"]
    return pos

# ── axis limits from LKF data ──────────────────────────────────────────────────
SUBSAMPLE = 10
all_vals = np.vstack([
    get_positions_filtered(lkf, f) for f in range(0, len(lkf), SUBSAMPLE)
])
margin = 0.5
xlim = (all_vals[:,0].min() - margin, all_vals[:,0].max() + margin)
ylim = (all_vals[:,1].min() - margin, all_vals[:,1].max() + margin)
zlim = (all_vals[:,2].min() - margin, all_vals[:,2].max() + margin)

# ── colours ────────────────────────────────────────────────────────────────────
C_NOISY = "#F59E0B"   # amber  — raw measured (noisy)
C_LKF   = "#2563EB"   # blue   — LKF filtered
C_EKF   = "#DC2626"   # red    — EKF filtered

# ── drawing helpers ────────────────────────────────────────────────────────────
def draw_skeleton(ax, pos, color, alpha=1.0, lw=1.8, dot_size=10):
    for pa, ch in BONES:
        if pa in JI and ch in JI:
            i, j = JI[pa], JI[ch]
            ax.plot([pos[i,0], pos[j,0]],
                    [pos[i,1], pos[j,1]],
                    [pos[i,2], pos[j,2]],
                    color=color, alpha=alpha, lw=lw)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2],
               c=color, s=dot_size, alpha=alpha, depthshade=False)

def setup_ax(ax, title, title_color="white"):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X (m)", fontsize=7, labelpad=1)
    ax.set_ylabel("Y (m)", fontsize=7, labelpad=1)
    ax.set_zlabel("Z (m)", fontsize=7, labelpad=1)
    ax.set_title(title, fontsize=10, color=title_color, pad=5, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.view_init(elev=15, azim=45)
    ax.set_facecolor("#1F2937")

# ── animation parameters ───────────────────────────────────────────────────────
TOTAL_FRAMES = len(lkf)
STEP   = 3
FRAMES = list(range(0, TOTAL_FRAMES, STEP))
FPS    = 30

# ── figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 6))
fig.patch.set_facecolor("#111827")

ax1 = fig.add_subplot(131, projection='3d')   # Measured
ax2 = fig.add_subplot(132, projection='3d')   # LKF
ax3 = fig.add_subplot(133, projection='3d')   # EKF

time_text = fig.text(0.5, 0.97, '', ha='center', va='top',
                     color='white', fontsize=11, family='monospace')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color=C_NOISY, lw=2, label='Measured (noisy)'),
    Line2D([0],[0], color=C_LKF,   lw=2, label='LKF (assembly)'),
    Line2D([0],[0], color=C_EKF,   lw=2, label='EKF (assembly)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           facecolor='#1F2937', edgecolor='gray',
           labelcolor='white', fontsize=9, framealpha=0.8,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])

# ── animation functions ────────────────────────────────────────────────────────
def init():
    return []

def animate(fi):
    frame = FRAMES[fi]
    t = frame * 0.01

    for ax in [ax1, ax2, ax3]:
        ax.cla()

    setup_ax(ax1, "Panel 1 — Measured (noisy)", title_color=C_NOISY)
    setup_ax(ax2, "Panel 2 — LKF (assembly)",   title_color=C_LKF)
    setup_ax(ax3, "Panel 3 — EKF (assembly)",   title_color=C_EKF)

    # Panel 1: raw noisy measurements from noisy.csv
    noisy_pos = get_positions_noisy(noisy_raw, frame)
    draw_skeleton(ax1, noisy_pos, C_NOISY, alpha=0.9, lw=1.5, dot_size=8)

    # Panel 2: LKF assembly posterior position estimates
    lkf_pos = get_positions_filtered(lkf, frame)
    draw_skeleton(ax2, lkf_pos, C_LKF, alpha=1.0, lw=2.0, dot_size=10)

    # Panel 3: EKF assembly posterior position estimates
    ekf_pos = get_positions_filtered(ekf, frame)
    draw_skeleton(ax3, ekf_pos, C_EKF, alpha=1.0, lw=2.0, dot_size=10)

    time_text.set_text(
        f"t = {t:6.2f} s   |   frame {frame:04d} / {TOTAL_FRAMES}"
    )
    return []

# ── render ─────────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, animate,
    frames=len(FRAMES),
    init_func=init,
    interval=1000 / FPS,
    blit=False
)

out_mp4 = os.path.join(OUTPUT, "skeleton_animation.mp4")
writer  = animation.FFMpegWriter(
    fps=FPS, bitrate=2200,
    extra_args=['-vcodec','libx264','-pix_fmt','yuv420p','-crf','20']
)

print(f"Rendering {len(FRAMES)} frames → {out_mp4} ...")
ani.save(out_mp4, writer=writer, dpi=100)
plt.close()
print(f"  ✓ Saved: {out_mp4}")
