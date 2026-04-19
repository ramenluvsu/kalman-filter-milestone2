#!/usr/bin/env python3
"""
verify.py — Milestone 3 Numerical Verification
Compares assembly output CSVs against C++ reference outputs.
Reports per-joint and per-state absolute errors.
Checks all errors <= 1e-9 as required by Section 6 of the spec.

Usage:
    python3 verify.py lkf_asm.csv lkf_cpp.csv ekf_asm.csv ekf_cpp.csv
"""

import sys
import csv
import math

TOL = 1e-9  # tolerance from spec Section 6

JOINT_NAMES = [
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
]
STATE_NAMES = ["px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"]

N_JOINTS = 23
N_STATE  = 12


def load_csv(path):
    """Load CSV into list of lists of floats (skip header)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                rows.append([float(v) for v in row])
    return rows


def verify(asm_path, cpp_path, label):
    """
    Compare two output CSVs.
    Returns True if all errors <= TOL.
    Prints a full verification table.
    """
    print(f"\n{'='*70}")
    print(f"  {label} Verification")
    print(f"  ASM : {asm_path}")
    print(f"  C++ : {cpp_path}")
    print(f"{'='*70}")

    asm_data = load_csv(asm_path)
    cpp_data = load_csv(cpp_path)

    if len(asm_data) != len(cpp_data):
        print(f"  ERROR: frame count mismatch "
              f"(asm={len(asm_data)}, cpp={len(cpp_data)})")
        return False

    T = len(asm_data)
    total_cols = N_JOINTS * N_STATE

    # per_joint_errors[jt] = list of abs errors across all frames and states
    per_joint_errors  = [[] for _ in range(N_JOINTS)]
    # per_state_errors[s]  = list of abs errors across all frames and joints
    per_state_errors  = [[] for _ in range(N_STATE)]

    max_err = 0.0
    min_err = float('inf')
    all_pass = True

    for k in range(T):
        asm_row = asm_data[k]
        cpp_row = cpp_data[k]
        for jt in range(N_JOINTS):
            for s in range(N_STATE):
                col = jt * N_STATE + s
                err = abs(asm_row[col] - cpp_row[col])
                per_joint_errors[jt].append(err)
                per_state_errors[s].append(err)
                if err > max_err:
                    max_err = err
                if err < min_err:
                    min_err = err
                if err > TOL:
                    all_pass = False

    # ── Per-joint table ──────────────────────────────────────────────────────
    print(f"\n  Per-Joint Average Absolute Error (averaged over all frames and states):")
    print(f"  {'Joint':<20} {'Avg Error':>14} {'Max Error':>14} {'PASS?':>6}")
    print(f"  {'-'*58}")
    for jt in range(N_JOINTS):
        errs = per_joint_errors[jt]
        avg  = sum(errs) / len(errs)
        mx   = max(errs)
        ok   = "PASS" if mx <= TOL else "FAIL"
        print(f"  {JOINT_NAMES[jt]:<20} {avg:>14.3e} {mx:>14.3e} {ok:>6}")

    # ── Per-state table ──────────────────────────────────────────────────────
    print(f"\n  Per-State Average Absolute Error (averaged over all frames and joints):")
    print(f"  {'State':<8} {'Avg Error':>14} {'Max Error':>14} {'PASS?':>6}")
    print(f"  {'-'*46}")
    for s in range(N_STATE):
        errs = per_state_errors[s]
        avg  = sum(errs) / len(errs)
        mx   = max(errs)
        ok   = "PASS" if mx <= TOL else "FAIL"
        print(f"  {STATE_NAMES[s]:<8} {avg:>14.3e} {mx:>14.3e} {ok:>6}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  Summary:")
    print(f"    Frames checked : {T}")
    print(f"    Total values   : {T * total_cols}")
    print(f"    Max error      : {max_err:.6e}")
    print(f"    Min error      : {min_err:.6e}")
    print(f"    Tolerance      : {TOL:.6e}")
    print(f"    Result         : {'ALL PASS ✓' if all_pass else 'FAILED ✗'}")

    return all_pass


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 verify.py "
              "lkf_asm.csv lkf_cpp.csv ekf_asm.csv ekf_cpp.csv")
        sys.exit(1)

    lkf_asm = sys.argv[1]
    lkf_cpp = sys.argv[2]
    ekf_asm = sys.argv[3]
    ekf_cpp = sys.argv[4]

    lkf_ok = verify(lkf_asm, lkf_cpp, "LKF")
    ekf_ok = verify(ekf_asm, ekf_cpp, "EKF")

    print(f"\n{'='*70}")
    print(f"  FINAL RESULT")
    print(f"    LKF : {'PASS ✓' if lkf_ok else 'FAIL ✗'}")
    print(f"    EKF : {'PASS ✓' if ekf_ok else 'FAIL ✗'}")
    print(f"{'='*70}\n")

    sys.exit(0 if (lkf_ok and ekf_ok) else 1)


if __name__ == "__main__":
    main()