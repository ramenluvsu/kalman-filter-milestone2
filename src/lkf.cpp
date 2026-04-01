/*
 * Linear Kalman Filter (LKF) - Full Body 3D Human Gait
 * FIXED: Uses ONE large 276-dimensional state vector for all 23 joints
 * State: 276x1 (23 joints x 12 states each)
 * F: 276x276 block diagonal
 * Q: 276x276 block diagonal
 * H: 69x276 block diagonal
 * P: 276x276
 * Memory: all large matrices heap-allocated
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

// ─── dimensions ───────────────────────────────────────────────────────────────
static const int N_JOINTS  = 23;
static const int N_S       = 12;          // states per joint
static const int N_M       = 3;           // measurements per joint
static const int N_STATE   = N_JOINTS * N_S;   // 276 total states
static const int N_MEAS    = N_JOINTS * N_M;   // 69 total measurements
static const double DT     = 1.0 / 100.0;
static const double SIGMA_J2 = 1.0;
static const double R_NOISE  = 0.01;

// ─── matrix helpers ───────────────────────────────────────────────────────────
static void mat_zero(double* A, int r, int c) {
    for (int i = 0; i < r*c; i++) A[i] = 0.0;
}
static void mat_eye(double* A, int n) {
    mat_zero(A, n, n);
    for (int i = 0; i < n; i++) A[i*n+i] = 1.0;
}
static void mat_scale(double* A, double s, int r, int c) {
    for (int i = 0; i < r*c; i++) A[i] *= s;
}
// C = A*B  (A:r×k, B:k×c, C:r×c)
static void mat_mul(const double* A, const double* B, double* C,
                    int r, int k, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            double s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p]*B[p*c+j];
            C[i*c+j] = s;
        }
}
static void mat_add(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i]+B[i];
}
static void mat_sub(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i]-B[i];
}
static void mat_T(const double* A, double* B, int r, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            B[j*r+i] = A[i*c+j];
}

// Cholesky solve: A*x = b, A is n×n SPD
static void chol_solve(const double* A, const double* b, double* x, int n) {
    std::vector<double> L(n*n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[i*n+j];
            for (int k = 0; k < j; k++) s -= L[i*n+k]*L[j*n+k];
            if (i == j) {
                if (s <= 1e-15) s = 1e-15;
                L[i*n+j] = std::sqrt(s);
            } else {
                L[i*n+j] = s / L[j*n+j];
            }
        }
    }
    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[i*n+k]*y[k];
        y[i] = s / L[i*n+i];
    }
    for (int i = n-1; i >= 0; i--) {
        double s = y[i];
        for (int k = i+1; k < n; k++) s -= L[k*n+i]*x[k];
        x[i] = s / L[i*n+i];
    }
}

// Kalman gain: K = P*Ht * S^{-1}
// K: N_STATE x N_MEAS, P: N_STATE x N_STATE, H: N_MEAS x N_STATE, S: N_MEAS x N_MEAS
static void compute_gain(const double* P, const double* H,
                          const double* S, double* K,
                          int ns, int nm) {
    // PHt = P * Ht  (ns x nm)
    std::vector<double> Ht(ns*nm), PHt(ns*nm);
    mat_T(H, Ht.data(), nm, ns);
    mat_mul(P, Ht.data(), PHt.data(), ns, ns, nm);
    // K[:,j] = PHt * S^{-1}[:,j]
    for (int j = 0; j < nm; j++) {
        std::vector<double> ej(nm, 0.0), Sinv_col(nm);
        ej[j] = 1.0;
        chol_solve(S, ej.data(), Sinv_col.data(), nm);
        for (int i = 0; i < ns; i++) {
            double s = 0;
            for (int k = 0; k < nm; k++) s += PHt[i*nm+k]*Sinv_col[k];
            K[i*nm+j] = s;
        }
    }
}

// ─── Build 276x276 block-diagonal F ──────────────────────────────────────────
static void build_F_full(double* F) {
    mat_zero(F, N_STATE, N_STATE);
    double dt=DT, dt2=dt*dt/2.0, dt3=dt*dt*dt/6.0;
    // Each joint gets a 12x12 block at offset jt*12
    for (int jt = 0; jt < N_JOINTS; jt++) {
        int o = jt * N_S;
        // 3 identical 4x4 axis blocks within this joint block
        for (int b = 0; b < 3; b++) {
            int r = o + b*4;
            F[(r+0)*N_STATE+(r+0)] = 1;   F[(r+0)*N_STATE+(r+1)] = dt;
            F[(r+0)*N_STATE+(r+2)] = dt2; F[(r+0)*N_STATE+(r+3)] = dt3;
            F[(r+1)*N_STATE+(r+1)] = 1;   F[(r+1)*N_STATE+(r+2)] = dt;
            F[(r+1)*N_STATE+(r+3)] = dt*dt/2.0;
            F[(r+2)*N_STATE+(r+2)] = 1;   F[(r+2)*N_STATE+(r+3)] = dt;
            F[(r+3)*N_STATE+(r+3)] = 1;
        }
    }
}

// ─── Build 276x276 block-diagonal Q ──────────────────────────────────────────
static void build_Q_full(double* Q) {
    mat_zero(Q, N_STATE, N_STATE);
    double dt=DT;
    double dt2=dt*dt, dt3=dt2*dt, dt4=dt3*dt, dt5=dt4*dt, dt6=dt5*dt;
    double q[4][4] = {
        {dt6/36, dt5/12, dt4/6,  dt3/6},
        {dt5/12, dt4/4,  dt3/2,  dt2/2},
        {dt4/6,  dt3/2,  dt2,    dt   },
        {dt3/6,  dt2/2,  dt,     1.0  }
    };
    for (int jt = 0; jt < N_JOINTS; jt++) {
        int o = jt * N_S;
        for (int b = 0; b < 3; b++) {
            int r = o + b*4;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    Q[(r+i)*N_STATE+(r+j)] = SIGMA_J2 * q[i][j];
        }
    }
}

// ─── Build 69x276 block-diagonal H ───────────────────────────────────────────
static void build_H_full(double* H) {
    mat_zero(H, N_MEAS, N_STATE);
    for (int jt = 0; jt < N_JOINTS; jt++) {
        int row = jt * N_M;   // measurement row offset
        int col = jt * N_S;   // state column offset
        H[(row+0)*N_STATE+(col+0)] = 1.0;  // px
        H[(row+1)*N_STATE+(col+4)] = 1.0;  // py
        H[(row+2)*N_STATE+(col+8)] = 1.0;  // pz
    }
}

// ─── Build 69x69 block-diagonal R ────────────────────────────────────────────
static void build_R_full(double* R) {
    mat_zero(R, N_MEAS, N_MEAS);
    for (int i = 0; i < N_MEAS; i++) R[i*N_MEAS+i] = R_NOISE;
}

// ─── CSV loader ───────────────────────────────────────────────────────────────
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

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "[LKF] Loading dataset..." << std::endl;
    auto noisy = load_csv("data/noisy.csv");
    int T = (int)noisy.size();
    std::cout << "[LKF] Frames: " << T
              << ", State dim: " << N_STATE
              << ", Meas dim: "  << N_MEAS << std::endl;

    // ── Build full system matrices (heap allocated) ───────────────────────────
    double* F  = new double[N_STATE*N_STATE];
    double* Q  = new double[N_STATE*N_STATE];
    double* H  = new double[N_MEAS *N_STATE];
    double* R  = new double[N_MEAS *N_MEAS ];
    double* Ft = new double[N_STATE*N_STATE];

    build_F_full(F);
    build_Q_full(Q);
    build_H_full(H);
    build_R_full(R);
    mat_T(F, Ft, N_STATE, N_STATE);

    // ── Allocate full state and covariance (heap) ─────────────────────────────
    double* x      = new double[N_STATE];
    double* P      = new double[N_STATE*N_STATE];
    double* x_pred = new double[N_STATE];
    double* P_pred = new double[N_STATE*N_STATE];

    // Working matrices
    double* FP    = new double[N_STATE*N_STATE];
    double* FPFt  = new double[N_STATE*N_STATE];
    double* HP    = new double[N_MEAS *N_STATE];
    double* HPHt  = new double[N_MEAS *N_MEAS ];
    double* Ht    = new double[N_STATE*N_MEAS ];
    double* S     = new double[N_MEAS *N_MEAS ];
    double* K     = new double[N_STATE*N_MEAS ];
    double* Ky    = new double[N_STATE];
    double* Hx    = new double[N_MEAS ];
    double* y     = new double[N_MEAS ];
    double* KH    = new double[N_STATE*N_STATE];
    double* IKH   = new double[N_STATE*N_STATE];
    double* IKHt  = new double[N_STATE*N_STATE];
    double* tmp1  = new double[N_STATE*N_STATE];
    double* tmp2  = new double[N_STATE*N_STATE];
    double* KR    = new double[N_STATE*N_MEAS ];
    double* KRKt  = new double[N_STATE*N_STATE];
    double* Kt    = new double[N_MEAS *N_STATE];
    double* I_big = new double[N_STATE*N_STATE];
    mat_eye(I_big, N_STATE);
    mat_T(H, Ht, N_MEAS, N_STATE);

    // ── Initialise full state vector x (276x1) ────────────────────────────────
    mat_zero(x, N_STATE, 1);
    for (int jt = 0; jt < N_JOINTS; jt++) {
        x[jt*N_S+0] = noisy[0][jt*3+0];  // px
        x[jt*N_S+4] = noisy[0][jt*3+1];  // py
        x[jt*N_S+8] = noisy[0][jt*3+2];  // pz
    }

    // ── Initialise P (276x276) = 100*I ───────────────────────────────────────
    mat_eye(P, N_STATE);
    mat_scale(P, 100.0, N_STATE, N_STATE);

    // ── Output storage ────────────────────────────────────────────────────────
    std::vector<std::vector<double>> output(T, std::vector<double>(N_STATE));
    for (int s = 0; s < N_STATE; s++) output[0][s] = x[s];

    // ── Main filter loop ──────────────────────────────────────────────────────
    for (int k = 1; k < T; k++) {

        // PREDICT
        mat_mul(F, x, x_pred, N_STATE, N_STATE, 1);
        mat_mul(F, P, FP, N_STATE, N_STATE, N_STATE);
        mat_mul(FP, Ft, FPFt, N_STATE, N_STATE, N_STATE);
        mat_add(FPFt, Q, P_pred, N_STATE*N_STATE);

        // BUILD FULL MEASUREMENT VECTOR z (69x1)
        double z[N_MEAS];
        for (int jt = 0; jt < N_JOINTS; jt++) {
            z[jt*3+0] = noisy[k][jt*3+0];
            z[jt*3+1] = noisy[k][jt*3+1];
            z[jt*3+2] = noisy[k][jt*3+2];
        }

        // INNOVATION y = z - H*x_pred
        mat_mul(H, x_pred, Hx, N_MEAS, N_STATE, 1);
        mat_sub(z, Hx, y, N_MEAS);

        // INNOVATION COVARIANCE S = H*P_pred*Ht + R
        mat_mul(H, P_pred, HP, N_MEAS, N_STATE, N_STATE);
        mat_mul(HP, Ht, HPHt, N_MEAS, N_STATE, N_MEAS);
        mat_add(HPHt, R, S, N_MEAS*N_MEAS);

        // KALMAN GAIN K (via Cholesky, no direct inversion)
        compute_gain(P_pred, H, S, K, N_STATE, N_MEAS);

        // STATE UPDATE x = x_pred + K*y
        mat_mul(K, y, Ky, N_STATE, N_MEAS, 1);
        mat_add(x_pred, Ky, x, N_STATE);

        // COVARIANCE UPDATE (Joseph form for numerical stability)
        mat_mul(K, H, KH, N_STATE, N_MEAS, N_STATE);
        mat_sub(I_big, KH, IKH, N_STATE*N_STATE);
        mat_mul(IKH, P_pred, tmp1, N_STATE, N_STATE, N_STATE);
        mat_T(IKH, IKHt, N_STATE, N_STATE);
        mat_mul(tmp1, IKHt, tmp2, N_STATE, N_STATE, N_STATE);
        mat_mul(K, R, KR, N_STATE, N_MEAS, N_MEAS);
        mat_T(K, Kt, N_STATE, N_MEAS);
        mat_mul(KR, Kt, KRKt, N_STATE, N_MEAS, N_STATE);
        mat_add(tmp2, KRKt, P, N_STATE*N_STATE);

        for (int s = 0; s < N_STATE; s++) output[k][s] = x[s];

        if (k % 500 == 0)
            std::cout << "[LKF] Frame " << k << "/" << T << std::endl;
    }

    // ── Write output CSV ──────────────────────────────────────────────────────
    std::ofstream out("output/lkf_output.csv");
    std::vector<std::string> joints = {
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    std::vector<std::string> states = {
        "px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"
    };
    bool first = true;
    for (auto& jn : joints) for (auto& st : states) {
        if (!first) out << ",";
        out << jn << "_" << st;
        first = false;
    }
    out << "\n";
    for (int k = 0; k < T; k++) {
        for (int i = 0; i < N_STATE; i++) {
            if (i > 0) out << ",";
            out << output[k][i];
        }
        out << "\n";
    }
    out.close();
    std::cout << "[LKF] Done. Output: output/lkf_output.csv" << std::endl;

    // Cleanup
    delete[] F; delete[] Q; delete[] H; delete[] R; delete[] Ft;
    delete[] x; delete[] P; delete[] x_pred; delete[] P_pred;
    delete[] FP; delete[] FPFt; delete[] HP; delete[] HPHt;
    delete[] Ht; delete[] S; delete[] K; delete[] Ky;
    delete[] Hx; delete[] y; delete[] KH; delete[] IKH;
    delete[] IKHt; delete[] tmp1; delete[] tmp2;
    delete[] KR; delete[] KRKt; delete[] Kt; delete[] I_big;

    return 0;
}
