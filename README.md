# Kalman Filter — Milestone 2
**Branch:** `milestone-2`

## Overview
Linear Kalman Filter (LKF) and Extended Kalman Filter (EKF) implemented in C/C++ for 3D full-body human gait estimation across 23 joints.

- State: 12 per joint (position, velocity, acceleration, jerk in x/y/z)
- Total state: 276 dimensions (23 × 12)
- Dataset: 3040 frames at 100 Hz

## Files
| File | Description |
|------|-------------|
| `src/lkf.cpp` | Linear Kalman Filter — Cartesian measurement model |
| `src/ekf.cpp` | Extended Kalman Filter — spherical measurement model |
| `plots/generate_plots.py` | Generates all required time-series and comparison plots |
| `simulation/animate.py` | 3D full-body walking animation (3 side-by-side views) |
| `report/report.tex` | Full LaTeX report |
| `Makefile` | Build both filters |

## Build & Run
```bash
# Place your CSV files as:
#   data/true.csv
#   data/noisy.csv

make all        # compiles lkf and ekf
./lkf           # runs LKF, writes output/lkf_output.csv
./ekf           # runs EKF, writes output/ekf_output.csv

cd plots && python generate_plots.py        # plots
cd simulation && python animate.py          # 3D animation
```

## Design Decisions
- **Matrix inversion**: Cholesky decomposition (not direct inversion) for numerical stability
- **Memory**: All large matrices heap-allocated with `new`/`delete`
- **Manual arctan2**: Piecewise polynomial approximation, max error ~0.0038 rad
- **Efficiency**: Per-joint processing avoids 276×276 matrix operations
