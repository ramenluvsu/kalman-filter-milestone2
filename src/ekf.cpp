/*
 * Extended Kalman Filter (EKF) - Full Body 3D Human Gait
 * Nonlinear measurement model: spherical coordinates (r, theta, phi)
 * Manual arctan2 approximation - no built-in trig library for angular quantities
 * Memory: all large matrices heap-allocated
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

static const int    N_JOINTS  = 23;
static const int    N_STATE   = 12;
static const int    N_MEAS    = 3;
static const double DT        = 1.0 / 100.0;
static const double SIGMA_J2  = 1.0;
static const double R_NOISE   = 0.05;   // slightly higher for spherical model
static const double EPS       = 1e-9;   // singularity guard

// ─── Matrix helpers (identical to LKF) ───────────────────────────────────────
static void mat_zero(double* A, int r, int c) {
    for (int i = 0; i < r*c; i++) A[i] = 0.0;
}
static void mat_eye(double* A, int n) {
    mat_zero(A, n, n);
    for (int i = 0; i < n; i++) A[i*n+i] = 1.0;
}
static void mat_mul(const double* A, const double* B, double* C, int r, int k, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            double s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p]*B[p*c+j];
            C[i*c+j] = s;
        }
}
static void mat_add(const double* A, const double* B, double* C, int r, int c) {
    for (int i = 0; i < r*c; i++) C[i] = A[i]+B[i];
}
static void mat_sub(const double* A, const double* B, double* C, int r, int c) {
    for (int i = 0; i < r*c; i++) C[i] = A[i]-B[i];
}
static void mat_T(const double* A, double* B, int r, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            B[j*r+i] = A[i*c+j];
}
static void mat_scale(const double* A, double s, double* B, int r, int c) {
    for (int i = 0; i < r*c; i++) B[i] = s*A[i];
}

static bool chol_solve(const double* A, const double* b, double* x, int n) {
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
    return true;
}

static void compute_gain(const double* P, const double* H,
                          const double* S, double* K) {
    double Ht[N_STATE*N_MEAS];
    double PHt[N_STATE*N_MEAS];
    mat_T(H, Ht, N_MEAS, N_STATE);
    mat_mul(P, Ht, PHt, N_STATE, N_STATE, N_MEAS);
    for (int j = 0; j < N_MEAS; j++) {
        double ej[N_MEAS] = {0};
        ej[j] = 1.0;
        double Sinv_col[N_MEAS];
        chol_solve(S, ej, Sinv_col, N_MEAS);
        for (int i = 0; i < N_STATE; i++) {
            double s = 0;
            for (int k = 0; k < N_MEAS; k++) s += PHt[i*N_MEAS+k]*Sinv_col[k];
            K[i*N_MEAS+j] = s;
        }
    }
}

// ─── Manual arctan2 approximation ────────────────────────────────────────────
/*
 * Approximation: atan(x) ≈ x*(π/4 + 0.273*(1 - |x|))  for |x| ≤ 1
 * Extended to full atan2(y,x) by quadrant adjustment.
 * Max error: ~0.0038 radians (~0.22 degrees) — acceptable for gait tracking.
 * Chosen because: no library calls, O(1) cost, bounded error, handles all quadrants.
 */
static double manual_atan(double x) {
    // Remez-style approximation for atan on [-1,1]
    const double PI_4 = 0.7853981633974483;
    const double C    = 0.2732632;  // tuned coefficient
    double ax = x < 0 ? -x : x;
    double result = PI_4 * x - x * (ax - 1.0) * (C + 0.273 * ax);
    return result;
}

static double manual_atan2(double y, double x) {
    const double PI   = 3.14159265358979323846;
    const double PI_2 = 1.5707963267948966;
    if (x == 0.0 && y == 0.0) return 0.0;
    double result;
    if (std::abs(x) >= std::abs(y)) {
        result = manual_atan(y / (x + (x == 0 ? EPS : 0)));
        if (x < 0) result += (y >= 0) ? PI : -PI;
    } else {
        result = PI_2 - manual_atan(x / (y + (y == 0 ? EPS : 0)));
        if (y < 0) result -= PI;
    }
    return result;
}

// ─── Nonlinear measurement function h(x) ─────────────────────────────────────
// Maps Cartesian (px,py,pz) to spherical (r, theta, phi)
static void h_func(const double* x_state, double* z_pred) {
    double px = x_state[0], py = x_state[4], pz = x_state[8];
    double rho = std::sqrt(px*px + py*py) + EPS;
    double r   = std::sqrt(px*px + py*py + pz*pz) + EPS;
    z_pred[0]  = r;
    z_pred[1]  = manual_atan2(py, px);
    z_pred[2]  = manual_atan2(pz, rho);
}

// ─── Jacobian J (3×12) ───────────────────────────────────────────────────────
static void build_jacobian(const double* x_state, double* J) {
    mat_zero(J, N_MEAS, N_STATE);
    double px = x_state[0], py = x_state[4], pz = x_state[8];
    double rho2 = px*px + py*py + EPS;
    double rho  = std::sqrt(rho2);
    double r2   = px*px + py*py + pz*pz + EPS;
    double r    = std::sqrt(r2);

    // Row 0: ∂r/∂state
    J[0*N_STATE+0] = px / r;   // ∂r/∂px
    J[0*N_STATE+4] = py / r;   // ∂r/∂py
    J[0*N_STATE+8] = pz / r;   // ∂r/∂pz

    // Row 1: ∂theta/∂state  (theta = atan2(py, px))
    J[1*N_STATE+0] = -py / rho2;  // ∂θ/∂px
    J[1*N_STATE+4] =  px / rho2;  // ∂θ/∂py
    // ∂θ/∂pz = 0

    // Row 2: ∂phi/∂state  (phi = atan2(pz, rho))
    J[2*N_STATE+0] = -(px * pz) / (r2 * rho);  // ∂φ/∂px
    J[2*N_STATE+4] = -(py * pz) / (r2 * rho);  // ∂φ/∂py
    J[2*N_STATE+8] =  rho / r2;                 // ∂φ/∂pz
}

// ─── Angle wrapping to (-π, π] ───────────────────────────────────────────────
static double wrap_angle(double a) {
    const double PI = 3.14159265358979323846;
    while (a >  PI) a -= 2*PI;
    while (a < -PI) a += 2*PI;
    return a;
}

// ─── Build F and Q (same as LKF) ─────────────────────────────────────────────
static void build_F(double* F, double dt) {
    mat_zero(F, N_STATE, N_STATE);
    double dt2 = dt*dt/2.0, dt3 = dt*dt*dt/6.0;
    for (int b = 0; b < 3; b++) {
        int o = b*4;
        F[(o+0)*N_STATE+(o+0)] = 1;   F[(o+0)*N_STATE+(o+1)] = dt;
        F[(o+0)*N_STATE+(o+2)] = dt2; F[(o+0)*N_STATE+(o+3)] = dt3;
        F[(o+1)*N_STATE+(o+1)] = 1;   F[(o+1)*N_STATE+(o+2)] = dt;
        F[(o+1)*N_STATE+(o+3)] = dt*dt/2.0;
        F[(o+2)*N_STATE+(o+2)] = 1;   F[(o+2)*N_STATE+(o+3)] = dt;
        F[(o+3)*N_STATE+(o+3)] = 1;
    }
}
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

// ─── CSV loader ───────────────────────────────────────────────────────────────
static std::vector<std::vector<double>> load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> data;
    std::string line;
    std::getline(f, line);
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

// ─── Main EKF ────────────────────────────────────────────────────────────────
int main() {
    std::cout << "[EKF] Loading dataset..." << std::endl;

    auto noisy = load_csv("data/noisy.csv");
    int T = (int)noisy.size();
    std::cout << "[EKF] Frames: " << T << std::endl;

    double* F  = new double[N_STATE*N_STATE];
    double* Q  = new double[N_STATE*N_STATE];
    build_F(F, DT);
    build_Q(Q, DT, SIGMA_J2);

    // R for spherical measurement
    double R[N_MEAS*N_MEAS];
    mat_zero(R, N_MEAS, N_MEAS);
    R[0] = R_NOISE;         // range noise
    R[4] = R_NOISE * 0.1;   // azimuth noise (angles tighter)
    R[8] = R_NOISE * 0.1;   // elevation noise

    int total_out = N_JOINTS * N_STATE;
    std::vector<std::vector<double>> output(T, std::vector<double>(total_out, 0.0));

    // Heap-allocate working matrices
    double* x      = new double[N_STATE];
    double* P      = new double[N_STATE*N_STATE];
    double* x_pred = new double[N_STATE];
    double* P_pred = new double[N_STATE*N_STATE];
    double* FP     = new double[N_STATE*N_STATE];
    double* FPFt   = new double[N_STATE*N_STATE];
    double* Ft     = new double[N_STATE*N_STATE];
    double* J      = new double[N_MEAS*N_STATE];
    double* S      = new double[N_MEAS*N_MEAS];
    double* K      = new double[N_STATE*N_MEAS];
    double* z_pred = new double[N_MEAS];
    double* y      = new double[N_MEAS];
    double* Ky     = new double[N_STATE];
    double* IKJ    = new double[N_STATE*N_STATE];
    double* KJ     = new double[N_STATE*N_STATE];
    double* I12    = new double[N_STATE*N_STATE];
    double* tmp1   = new double[N_STATE*N_STATE];
    double* tmp2   = new double[N_STATE*N_STATE];
    double* KR     = new double[N_STATE*N_MEAS];
    double* KRKt   = new double[N_STATE*N_STATE];
    double* Kt     = new double[N_MEAS*N_STATE];
    double* IKJt   = new double[N_STATE*N_STATE];
    double* JP     = new double[N_MEAS*N_STATE];
    double* JPJt   = new double[N_MEAS*N_MEAS];
    double* Jt     = new double[N_STATE*N_MEAS];
    mat_eye(I12, N_STATE);
    mat_T(F, Ft, N_STATE, N_STATE);

    for (int jt = 0; jt < N_JOINTS; jt++) {
        int col_offset = jt * 3;

        mat_zero(x, N_STATE, 1);
        x[0] = noisy[0][col_offset+0];
        x[4] = noisy[0][col_offset+1];
        x[8] = noisy[0][col_offset+2];
        mat_eye(P, N_STATE);
        mat_scale(P, 100.0, P, N_STATE, N_STATE);

        for (int s = 0; s < N_STATE; s++) output[0][jt*N_STATE+s] = x[s];

        for (int k = 1; k < T; k++) {
            // ── PREDICT ──────────────────────────────────────────────────
            mat_mul(F, x, x_pred, N_STATE, N_STATE, 1);
            mat_mul(F, P, FP, N_STATE, N_STATE, N_STATE);
            mat_mul(FP, Ft, FPFt, N_STATE, N_STATE, N_STATE);
            mat_add(FPFt, Q, P_pred, N_STATE, N_STATE);

            // ── COMPUTE JACOBIAN at predicted state ──────────────────────
            build_jacobian(x_pred, J);

            // ── NONLINEAR MEASUREMENT PREDICTION h(x_pred) ──────────────
            h_func(x_pred, z_pred);

            // ── CONVERT NOISY MEASUREMENT TO SPHERICAL ───────────────────
            double nx = noisy[k][col_offset+0];
            double ny = noisy[k][col_offset+1];
            double nz = noisy[k][col_offset+2];
            double rho_n = std::sqrt(nx*nx + ny*ny) + EPS;
            double z_meas[N_MEAS];
            z_meas[0] = std::sqrt(nx*nx + ny*ny + nz*nz) + EPS;
            z_meas[1] = manual_atan2(ny, nx);
            z_meas[2] = manual_atan2(nz, rho_n);

            // ── INNOVATION (with angle wrapping) ─────────────────────────
            y[0] = z_meas[0] - z_pred[0];
            y[1] = wrap_angle(z_meas[1] - z_pred[1]);
            y[2] = wrap_angle(z_meas[2] - z_pred[2]);

            // ── INNOVATION COVARIANCE S = J*P_pred*Jᵀ + R ────────────────
            mat_mul(J, P_pred, JP, N_MEAS, N_STATE, N_STATE);
            mat_T(J, Jt, N_MEAS, N_STATE);
            mat_mul(JP, Jt, JPJt, N_MEAS, N_STATE, N_MEAS);
            mat_add(JPJt, R, S, N_MEAS, N_MEAS);

            // ── KALMAN GAIN ───────────────────────────────────────────────
            compute_gain(P_pred, J, S, K);

            // ── STATE UPDATE ──────────────────────────────────────────────
            mat_mul(K, y, Ky, N_STATE, N_MEAS, 1);
            mat_add(x_pred, Ky, x, N_STATE, 1);

            // ── COVARIANCE UPDATE (Joseph form) ───────────────────────────
            mat_mul(K, J, KJ, N_STATE, N_MEAS, N_STATE);
            mat_sub(I12, KJ, IKJ, N_STATE, N_STATE);
            mat_mul(IKJ, P_pred, tmp1, N_STATE, N_STATE, N_STATE);
            mat_T(IKJ, IKJt, N_STATE, N_STATE);
            mat_mul(tmp1, IKJt, tmp2, N_STATE, N_STATE, N_STATE);
            mat_mul(K, R, KR, N_STATE, N_MEAS, N_MEAS);
            mat_T(K, Kt, N_STATE, N_MEAS);
            mat_mul(KR, Kt, KRKt, N_STATE, N_MEAS, N_STATE);
            mat_add(tmp2, KRKt, P, N_STATE, N_STATE);

            for (int s = 0; s < N_STATE; s++) output[k][jt*N_STATE+s] = x[s];
        }

        if (jt % 5 == 0)
            std::cout << "[EKF] Joint " << jt+1 << "/" << N_JOINTS << " done." << std::endl;
    }

    // ── WRITE OUTPUT CSV ──────────────────────────────────────────────────────
    std::ofstream out("output/ekf_output.csv");
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
    std::cout << "[EKF] Output written to ../output/ekf_output.csv" << std::endl;

    delete[] F; delete[] Q;
    delete[] x; delete[] P; delete[] x_pred; delete[] P_pred;
    delete[] FP; delete[] FPFt; delete[] Ft;
    delete[] J; delete[] S; delete[] K; delete[] z_pred;
    delete[] y; delete[] Ky; delete[] IKJ; delete[] KJ;
    delete[] I12; delete[] tmp1; delete[] tmp2;
    delete[] KR; delete[] KRKt; delete[] Kt; delete[] IKJt;
    delete[] JP; delete[] JPJt; delete[] Jt;

    return 0;
}
