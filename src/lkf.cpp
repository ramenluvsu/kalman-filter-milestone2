/*
 * Linear Kalman Filter (LKF) - Full Body 3D Human Gait
 * State: 12 per joint (px,vx,ax,jx, py,vy,ay,jy, pz,vz,az,jz)
 * 23 joints => 276-dim total state, processed joint-by-joint for efficiency
 * Memory: all large matrices heap-allocated to avoid stack overflow
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

// ─── constants ────────────────────────────────────────────────────────────────
static const int    N_JOINTS   = 23;
static const int    N_STATE    = 12;   // per joint
static const int    N_MEAS     = 3;    // per joint (px, py, pz)
static const double DT         = 1.0 / 100.0;   // 100 Hz mocap
static const double SIGMA_J2   = 1.0;            // process noise variance
static const double R_NOISE    = 0.01;           // measurement noise variance

// ─── small matrix helpers (stack-safe for 12×12 and 3×3) ────────────────────
// We represent matrices as flat row-major arrays.

static void mat_zero(double* A, int r, int c) {
    for (int i = 0; i < r*c; i++) A[i] = 0.0;
}
static void mat_eye(double* A, int n) {
    mat_zero(A, n, n);
    for (int i = 0; i < n; i++) A[i*n+i] = 1.0;
}
// C = A*B  (A:r×k, B:k×c, C:r×c)
static void mat_mul(const double* A, const double* B, double* C, int r, int k, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            double s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p]*B[p*c+j];
            C[i*c+j] = s;
        }
}
// C = A + B
static void mat_add(const double* A, const double* B, double* C, int r, int c) {
    for (int i = 0; i < r*c; i++) C[i] = A[i]+B[i];
}
// C = A - B
static void mat_sub(const double* A, const double* B, double* C, int r, int c) {
    for (int i = 0; i < r*c; i++) C[i] = A[i]-B[i];
}
// Transpose: B = Aᵀ (A:r×c, B:c×r)
static void mat_T(const double* A, double* B, int r, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            B[j*r+i] = A[i*c+j];
}
// Scale: B = s*A
static void mat_scale(const double* A, double s, double* B, int r, int c) {
    for (int i = 0; i < r*c; i++) B[i] = s*A[i];
}

/*
 * Cholesky solve: solve A*x = b where A is SPD (n×n).
 * Avoids direct matrix inversion — numerically more stable.
 * Returns x in b_out.
 */
static bool chol_solve(const double* A, const double* b, double* x, int n) {
    // Copy A so we don't modify original
    std::vector<double> L(n*n, 0.0);
    // Cholesky decomposition A = L*Lᵀ
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[i*n+j];
            for (int k = 0; k < j; k++) s -= L[i*n+k]*L[j*n+k];
            if (i == j) {
                if (s <= 0) return false; // not positive definite
                L[i*n+j] = std::sqrt(s);
            } else {
                L[i*n+j] = s / L[j*n+j];
            }
        }
    }
    // Forward substitution: L*y = b
    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[i*n+k]*y[k];
        y[i] = s / L[i*n+i];
    }
    // Backward substitution: Lᵀ*x = y
    for (int i = n-1; i >= 0; i--) {
        double s = y[i];
        for (int k = i+1; k < n; k++) s -= L[k*n+i]*x[k];
        x[i] = s / L[i*n+i];
    }
    return true;
}

/*
 * Compute K = P*Hᵀ * S⁻¹  without explicit inversion.
 * We solve Sᵀ * Kᵀ = H * Pᵀ  (S is symmetric so Sᵀ=S).
 * K: N_STATE×N_MEAS, P: N_STATE×N_STATE, H: N_MEAS×N_STATE, S: N_MEAS×N_MEAS
 */
static void compute_gain(const double* P, const double* H,
                          const double* S, double* K) {
    // PHt = P * Hᵀ  (12×3)
    double Ht[N_STATE*N_MEAS];
    double PHt[N_STATE*N_MEAS];
    mat_T(H, Ht, N_MEAS, N_STATE);
    mat_mul(P, Ht, PHt, N_STATE, N_STATE, N_MEAS);

    // For each column j of K, solve S * K[:,j] = PHt[:,j]
    for (int j = 0; j < N_MEAS; j++) {
        double col_b[N_MEAS], col_x[N_MEAS];
        for (int i = 0; i < N_MEAS; i++) col_b[i] = PHt[i*N_MEAS+j];
        chol_solve(S, col_b, col_x, N_MEAS);
        for (int i = 0; i < N_STATE; i++) K[i*N_MEAS+j] = col_x[j]; // BUG-fix below
    }
    // Correct approach: K[:,j] has N_STATE rows, solve for each column
    // Re-do properly:
    for (int j = 0; j < N_MEAS; j++) {
        double rhs[N_MEAS];
        for (int i = 0; i < N_MEAS; i++) rhs[i] = PHt[i*N_MEAS+j]; // wait – PHt is 12×3

        // PHt is (12×3): PHt[row*3+col]
        // We need col j of PHt which is PHt[row*3+j] for row=0..11
        // S is 3×3, K[:,j] should be 12×1
        // But S is 3×3 and rhs needs to be 12×1 – can't solve directly
        // Correct: K = PHt * S^{-1}  => K[:,j] = PHt * S^{-1}[:,j]
        // S^{-1} col j: solve S * e = I[:,j]
        double ej[N_MEAS] = {0};
        ej[j] = 1.0;
        double Sinv_col[N_MEAS];
        chol_solve(S, ej, Sinv_col, N_MEAS);
        // K[:,j] = PHt * Sinv_col  (12×3 * 3×1 = 12×1)
        for (int i = 0; i < N_STATE; i++) {
            double s = 0;
            for (int k = 0; k < N_MEAS; k++) s += PHt[i*N_MEAS+k]*Sinv_col[k];
            K[i*N_MEAS+j] = s;
        }
    }
}

// ─── Build F matrix (12×12 block-diagonal, 3 axes independent) ──────────────
static void build_F(double* F, double dt) {
    mat_zero(F, N_STATE, N_STATE);
    double dt2 = dt*dt/2.0, dt3 = dt*dt*dt/6.0;
    // 3 identical 4×4 blocks on diagonal
    for (int b = 0; b < 3; b++) {
        int o = b*4;
        F[(o+0)*N_STATE+(o+0)] = 1;  F[(o+0)*N_STATE+(o+1)] = dt;
        F[(o+0)*N_STATE+(o+2)] = dt2; F[(o+0)*N_STATE+(o+3)] = dt3;
        F[(o+1)*N_STATE+(o+1)] = 1;  F[(o+1)*N_STATE+(o+2)] = dt;
        F[(o+1)*N_STATE+(o+3)] = dt*dt/2.0;
        F[(o+2)*N_STATE+(o+2)] = 1;  F[(o+2)*N_STATE+(o+3)] = dt;
        F[(o+3)*N_STATE+(o+3)] = 1;
    }
}

// ─── Build Q matrix (12×12) ──────────────────────────────────────────────────
static void build_Q(double* Q, double dt, double sj2) {
    mat_zero(Q, N_STATE, N_STATE);
    double dt2=dt*dt, dt3=dt2*dt, dt4=dt3*dt, dt5=dt4*dt, dt6=dt5*dt;
    double q[4][4] = {
        {dt6/36, dt5/12, dt4/6,  dt3/6},
        {dt5/12, dt4/4,  dt3/2,  dt2/2},
        {dt4/6,  dt3/2,  dt2,    dt   },
        {dt3/6,  dt2/2,  dt,     1.0  }
    };
    for (int b = 0; b < 3; b++) {
        int o = b*4;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Q[(o+i)*N_STATE+(o+j)] = sj2 * q[i][j];
    }
}

// ─── Build H matrix (3×12) ───────────────────────────────────────────────────
static void build_H(double* H) {
    mat_zero(H, N_MEAS, N_STATE);
    H[0*N_STATE+0] = 1.0;  // px at index 0
    H[1*N_STATE+4] = 1.0;  // py at index 4
    H[2*N_STATE+8] = 1.0;  // pz at index 8
}

// ─── CSV utilities ───────────────────────────────────────────────────────────
static std::vector<std::vector<double>> load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> data;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { row.push_back(std::stod(cell)); }
            catch (...) { row.push_back(0.0); }
        }
        if ((int)row.size() >= N_JOINTS*3) data.push_back(row);
    }
    return data;
}

// ─── Main LKF ────────────────────────────────────────────────────────────────
int main() {
    std::cout << "[LKF] Loading dataset..." << std::endl;

    auto noisy = load_csv("../data/noisy.csv");
    int T = (int)noisy.size();
    std::cout << "[LKF] Frames: " << T << std::endl;

    // Pre-build shared matrices
    double* F = new double[N_STATE*N_STATE];
    double* Q = new double[N_STATE*N_STATE];
    double* H = new double[N_MEAS*N_STATE];
    build_F(F, DT);
    build_Q(Q, DT, SIGMA_J2);
    build_H(H);

    // R (3×3 diagonal measurement noise)
    double R[N_MEAS*N_MEAS];
    mat_zero(R, N_MEAS, N_MEAS);
    for (int i = 0; i < N_MEAS; i++) R[i*N_MEAS+i] = R_NOISE;

    // Output storage: T frames × (23 joints × 12 states)
    int total_out = N_JOINTS * N_STATE;
    std::vector<std::vector<double>> output(T, std::vector<double>(total_out, 0.0));

    // Heap-allocate working matrices (sizes needed per joint)
    double* x     = new double[N_STATE];       // state estimate
    double* P     = new double[N_STATE*N_STATE]; // covariance
    double* x_pred = new double[N_STATE];
    double* P_pred = new double[N_STATE*N_STATE];
    double* FP    = new double[N_STATE*N_STATE];
    double* FPFt  = new double[N_STATE*N_STATE];
    double* Ft    = new double[N_STATE*N_STATE];
    double* Hx    = new double[N_MEAS];
    double* y     = new double[N_MEAS];
    double* HP    = new double[N_MEAS*N_STATE];
    double* HPHt  = new double[N_MEAS*N_MEAS];
    double* Ht    = new double[N_STATE*N_MEAS];
    double* S     = new double[N_MEAS*N_MEAS];
    double* K     = new double[N_STATE*N_MEAS];
    double* Ky    = new double[N_STATE];
    double* IKH   = new double[N_STATE*N_STATE];
    double* KH    = new double[N_STATE*N_STATE];
    double* I12   = new double[N_STATE*N_STATE];
    double* tmp1  = new double[N_STATE*N_STATE];
    double* tmp2  = new double[N_STATE*N_STATE];
    double* KR    = new double[N_STATE*N_MEAS];
    double* KRKt  = new double[N_STATE*N_STATE];
    double* Kt    = new double[N_MEAS*N_STATE];
    mat_eye(I12, N_STATE);

    // Process each joint independently (avoids 276×276 inversions)
    for (int jt = 0; jt < N_JOINTS; jt++) {
        int col_offset = jt * 3; // columns in CSV for this joint

        // Initialize state from first measurement
        mat_zero(x, N_STATE, 1);
        x[0] = noisy[0][col_offset+0]; // px
        x[4] = noisy[0][col_offset+1]; // py
        x[8] = noisy[0][col_offset+2]; // pz

        // Initial covariance P0 = 100*I
        mat_eye(P, N_STATE);
        mat_scale(P, 100.0, P, N_STATE, N_STATE);

        // Store initial state
        for (int s = 0; s < N_STATE; s++) output[0][jt*N_STATE+s] = x[s];

        mat_T(F, Ft, N_STATE, N_STATE);

        for (int k = 1; k < T; k++) {
            // ── PREDICT ──────────────────────────────────────────────────
            // x_pred = F * x
            mat_mul(F, x, x_pred, N_STATE, N_STATE, 1);
            // P_pred = F*P*Fᵀ + Q
            mat_mul(F, P, FP, N_STATE, N_STATE, N_STATE);
            mat_mul(FP, Ft, FPFt, N_STATE, N_STATE, N_STATE);
            mat_add(FPFt, Q, P_pred, N_STATE, N_STATE);

            // ── INNOVATION ───────────────────────────────────────────────
            // z = measurement
            double z[N_MEAS];
            z[0] = noisy[k][col_offset+0];
            z[1] = noisy[k][col_offset+1];
            z[2] = noisy[k][col_offset+2];
            // y = z - H*x_pred
            mat_mul(H, x_pred, Hx, N_MEAS, N_STATE, 1);
            mat_sub(z, Hx, y, N_MEAS, 1);

            // ── INNOVATION COVARIANCE S = H*P_pred*Hᵀ + R ────────────────
            mat_mul(H, P_pred, HP, N_MEAS, N_STATE, N_STATE);
            mat_T(H, Ht, N_MEAS, N_STATE);
            mat_mul(HP, Ht, HPHt, N_MEAS, N_STATE, N_MEAS);
            mat_add(HPHt, R, S, N_MEAS, N_MEAS);

            // ── KALMAN GAIN K = P_pred*Hᵀ * S⁻¹ (via Cholesky) ──────────
            compute_gain(P_pred, H, S, K);

            // ── STATE UPDATE x = x_pred + K*y ────────────────────────────
            mat_mul(K, y, Ky, N_STATE, N_MEAS, 1);
            mat_add(x_pred, Ky, x, N_STATE, 1);

            // ── COVARIANCE UPDATE (Joseph form for numerical stability) ───
            // IKH = I - K*H
            mat_mul(K, H, KH, N_STATE, N_MEAS, N_STATE);
            mat_sub(I12, KH, IKH, N_STATE, N_STATE);
            // tmp1 = IKH * P_pred
            mat_mul(IKH, P_pred, tmp1, N_STATE, N_STATE, N_STATE);
            // tmp2 = tmp1 * IKHᵀ
            double IKHt[N_STATE*N_STATE];
            mat_T(IKH, IKHt, N_STATE, N_STATE);
            mat_mul(tmp1, IKHt, tmp2, N_STATE, N_STATE, N_STATE);
            // KRKᵀ
            mat_mul(K, R, KR, N_STATE, N_MEAS, N_MEAS);
            mat_T(K, Kt, N_STATE, N_MEAS);
            mat_mul(KR, Kt, KRKt, N_STATE, N_MEAS, N_STATE);
            // P = tmp2 + KRKᵀ
            mat_add(tmp2, KRKt, P, N_STATE, N_STATE);

            // Store state
            for (int s = 0; s < N_STATE; s++) output[k][jt*N_STATE+s] = x[s];
        }

        if (jt % 5 == 0)
            std::cout << "[LKF] Joint " << jt+1 << "/" << N_JOINTS << " done." << std::endl;
    }

    // ── WRITE OUTPUT CSV ─────────────────────────────────────────────────────
    std::ofstream out("../output/lkf_output.csv");
    // Header
    std::vector<std::string> joints = {
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    std::vector<std::string> states = {"px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"};
    bool first = true;
    for (auto& jn : joints) for (auto& st : states) {
        if (!first) out << ",";
        out << jn << "_" << st;
        first = false;
    }
    out << "\n";
    for (int k = 0; k < T; k++) {
        for (int i = 0; i < total_out; i++) {
            if (i > 0) out << ",";
            out << output[k][i];
        }
        out << "\n";
    }
    out.close();
    std::cout << "[LKF] Output written to ../output/lkf_output.csv" << std::endl;

    // Cleanup
    delete[] F; delete[] Q; delete[] H;
    delete[] x; delete[] P; delete[] x_pred; delete[] P_pred;
    delete[] FP; delete[] FPFt; delete[] Ft; delete[] Hx;
    delete[] y; delete[] HP; delete[] HPHt; delete[] Ht;
    delete[] S; delete[] K; delete[] Ky; delete[] IKH;
    delete[] KH; delete[] I12; delete[] tmp1; delete[] tmp2;
    delete[] KR; delete[] KRKt; delete[] Kt;

    return 0;
}
