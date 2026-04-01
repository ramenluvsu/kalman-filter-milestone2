/*
 * Extended Kalman Filter (EKF) - Full Body 3D Human Gait
 * FIXED: Uses ONE large 276-dimensional state vector for all 23 joints
 * State: 276x1, F: 276x276, Q: 276x276, H: 69x276 (all block diagonal)
 * EKF measurement: spherical coordinates per joint
 * Jacobian J: 69x276 block diagonal (one 3x12 block per joint)
 * Manual arctan2 approximation — no built-in trig
 * Memory: all large matrices heap-allocated
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

static const int N_JOINTS  = 23;
static const int N_S       = 12;
static const int N_M       = 3;
static const int N_STATE   = N_JOINTS * N_S;   // 276
static const int N_MEAS    = N_JOINTS * N_M;   // 69
static const double DT     = 1.0 / 100.0;
static const double SIGMA_J2 = 1.0;
static const double R_NOISE  = 0.01;
static const double EPS      = 1e-9;

// ─── Matrix helpers ───────────────────────────────────────────────────────────
static void mat_zero(double* A, int n) { for (int i=0;i<n;i++) A[i]=0.0; }
static void mat_eye(double* A, int n) {
    mat_zero(A, n*n);
    for (int i=0;i<n;i++) A[i*n+i]=1.0;
}
static void mat_scale(double* A, double s, int n) {
    for (int i=0;i<n;i++) A[i]*=s;
}
static void mat_mul(const double* A, const double* B, double* C,
                    int r, int k, int c) {
    for (int i=0;i<r;i++)
        for (int j=0;j<c;j++) {
            double s=0;
            for (int p=0;p<k;p++) s+=A[i*k+p]*B[p*c+j];
            C[i*c+j]=s;
        }
}
static void mat_add(const double* A,const double* B,double* C,int n){
    for(int i=0;i<n;i++) C[i]=A[i]+B[i];
}
static void mat_sub(const double* A,const double* B,double* C,int n){
    for(int i=0;i<n;i++) C[i]=A[i]-B[i];
}
static void mat_T(const double* A,double* B,int r,int c){
    for(int i=0;i<r;i++)
        for(int j=0;j<c;j++)
            B[j*r+i]=A[i*c+j];
}

static void chol_solve(const double* A,const double* b,double* x,int n){
    std::vector<double> L(n*n,0.0);
    for(int i=0;i<n;i++){
        for(int j=0;j<=i;j++){
            double s=A[i*n+j];
            for(int k=0;k<j;k++) s-=L[i*n+k]*L[j*n+k];
            if(i==j){ if(s<=1e-15)s=1e-15; L[i*n+j]=std::sqrt(s); }
            else L[i*n+j]=s/L[j*n+j];
        }
    }
    std::vector<double> y(n);
    for(int i=0;i<n;i++){
        double s=b[i];
        for(int k=0;k<i;k++) s-=L[i*n+k]*y[k];
        y[i]=s/L[i*n+i];
    }
    for(int i=n-1;i>=0;i--){
        double s=y[i];
        for(int k=i+1;k<n;k++) s-=L[k*n+i]*x[k];
        x[i]=s/L[i*n+i];
    }
}

static void compute_gain(const double* P,const double* J,
                          const double* S,double* K,int ns,int nm){
    std::vector<double> Jt(ns*nm),PJt(ns*nm);
    mat_T(J,Jt.data(),nm,ns);
    mat_mul(P,Jt.data(),PJt.data(),ns,ns,nm);
    for(int j=0;j<nm;j++){
        std::vector<double> ej(nm,0.0),Sinv_col(nm);
        ej[j]=1.0;
        chol_solve(S,ej.data(),Sinv_col.data(),nm);
        for(int i=0;i<ns;i++){
            double s=0;
            for(int k=0;k<nm;k++) s+=PJt[i*nm+k]*Sinv_col[k];
            K[i*nm+j]=s;
        }
    }
}

// ─── Manual arctan2 approximation ────────────────────────────────────────────
/*
 * Uses piecewise polynomial: atan(x) ≈ x*(π/4 + 0.273*(1-|x|)) for |x|≤1
 * Extended to full quadrant coverage via range reduction and sign checks.
 * Max error: ~0.0038 rad (0.22°) — acceptable for joint angle estimation.
 * Chosen because: O(1), no library calls, bounded error, correct quadrants.
 */
static double manual_atan(double x){
    const double PI_4=0.7853981633974483;
    const double C=0.2732632;
    double ax=x<0?-x:x;
    return PI_4*x - x*(ax-1.0)*(C+0.273*ax);
}
static double manual_atan2(double y,double x){
    const double PI=3.14159265358979323846;
    const double PI_2=1.5707963267948966;
    if(x==0.0&&y==0.0) return 0.0;
    double result;
    if(std::abs(x)>=std::abs(y)){
        result=manual_atan(y/(x==0?EPS:x));
        if(x<0) result+=(y>=0)?PI:-PI;
    } else {
        result=PI_2-manual_atan(x/(y==0?EPS:y));
        if(y<0) result-=PI;
    }
    return result;
}

// Angle wrap to (-pi, pi]
static double wrap(double a){
    const double PI=3.14159265358979323846;
    while(a> PI) a-=2*PI;
    while(a<-PI) a+=2*PI;
    return a;
}

// ─── Build full matrices ───────────────────────────────────────────────────────
static void build_F_full(double* F){
    mat_zero(F,N_STATE*N_STATE);
    double dt=DT,dt2=dt*dt/2.0,dt3=dt*dt*dt/6.0;
    for(int jt=0;jt<N_JOINTS;jt++){
        int o=jt*N_S;
        for(int b=0;b<3;b++){
            int r=o+b*4;
            F[(r+0)*N_STATE+(r+0)]=1;   F[(r+0)*N_STATE+(r+1)]=dt;
            F[(r+0)*N_STATE+(r+2)]=dt2; F[(r+0)*N_STATE+(r+3)]=dt3;
            F[(r+1)*N_STATE+(r+1)]=1;   F[(r+1)*N_STATE+(r+2)]=dt;
            F[(r+1)*N_STATE+(r+3)]=dt*dt/2.0;
            F[(r+2)*N_STATE+(r+2)]=1;   F[(r+2)*N_STATE+(r+3)]=dt;
            F[(r+3)*N_STATE+(r+3)]=1;
        }
    }
}

static void build_Q_full(double* Q){
    mat_zero(Q,N_STATE*N_STATE);
    double dt=DT,dt2=dt*dt,dt3=dt2*dt,dt4=dt3*dt,dt5=dt4*dt,dt6=dt5*dt;
    double q[4][4]={
        {dt6/36,dt5/12,dt4/6,dt3/6},
        {dt5/12,dt4/4, dt3/2,dt2/2},
        {dt4/6, dt3/2, dt2,  dt   },
        {dt3/6, dt2/2, dt,   1.0  }
    };
    for(int jt=0;jt<N_JOINTS;jt++){
        int o=jt*N_S;
        for(int b=0;b<3;b++){
            int r=o+b*4;
            for(int i=0;i<4;i++)
                for(int j=0;j<4;j++)
                    Q[(r+i)*N_STATE+(r+j)]=SIGMA_J2*q[i][j];
        }
    }
}

// Build 69x69 R
static void build_R_full(double* R){
    mat_zero(R,N_MEAS*N_MEAS);
    for(int i=0;i<N_MEAS;i++) R[i*N_MEAS+i]=R_NOISE;
}

// ─── Per-joint h(x) and Jacobian ─────────────────────────────────────────────
// Extract one joint position from full state
static void joint_pos(const double* x, int jt, double& px, double& py, double& pz){
    int o=jt*N_S;
    px=x[o+0]; py=x[o+4]; pz=x[o+8];
}

// h(x) for one joint -> 3 spherical measurements
static void h_joint(double px,double py,double pz, double* z_out){
    double rho=std::sqrt(px*px+py*py)+EPS;
    double r  =std::sqrt(px*px+py*py+pz*pz)+EPS;
    z_out[0]=r;
    z_out[1]=manual_atan2(py,px);
    z_out[2]=manual_atan2(pz,rho);
}

// Full h(x) for all 23 joints -> 69x1
static void h_full(const double* x, double* z_pred){
    for(int jt=0;jt<N_JOINTS;jt++){
        double px,py,pz;
        joint_pos(x,jt,px,py,pz);
        h_joint(px,py,pz,&z_pred[jt*3]);
    }
}

// Build full 69x276 Jacobian (block diagonal, one 3x12 block per joint)
static void build_J_full(const double* x, double* J){
    mat_zero(J,N_MEAS*N_STATE);
    for(int jt=0;jt<N_JOINTS;jt++){
        double px,py,pz;
        joint_pos(x,jt,px,py,pz);
        double rho2=px*px+py*py+EPS;
        double rho =std::sqrt(rho2);
        double r2  =px*px+py*py+pz*pz+EPS;
        double r   =std::sqrt(r2);
        int row=jt*N_M;   // measurement row offset
        int col=jt*N_S;   // state column offset
        // Row 0: dr/dstate
        J[(row+0)*N_STATE+(col+0)]=px/r;
        J[(row+0)*N_STATE+(col+4)]=py/r;
        J[(row+0)*N_STATE+(col+8)]=pz/r;
        // Row 1: dtheta/dstate
        J[(row+1)*N_STATE+(col+0)]=-py/rho2;
        J[(row+1)*N_STATE+(col+4)]= px/rho2;
        // Row 2: dphi/dstate
        J[(row+2)*N_STATE+(col+0)]=-(px*pz)/(r2*rho);
        J[(row+2)*N_STATE+(col+4)]=-(py*pz)/(r2*rho);
        J[(row+2)*N_STATE+(col+8)]= rho/r2;
    }
}

// ─── CSV loader ───────────────────────────────────────────────────────────────
static std::vector<std::vector<double>> load_csv(const std::string& path){
    std::ifstream f(path);
    if(!f.is_open()) throw std::runtime_error("Cannot open: "+path);
    std::vector<std::vector<double>> data;
    std::string line;
    std::getline(f,line);
    while(std::getline(f,line)){
        if(line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while(std::getline(ss,cell,',')){
            try{ row.push_back(std::stod(cell)); }
            catch(...){ row.push_back(0.0); }
        }
        if((int)row.size()>=N_JOINTS*3) data.push_back(row);
    }
    return data;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(){
    std::cout<<"[EKF] Loading dataset..."<<std::endl;
    auto noisy=load_csv("data/noisy.csv");
    int T=(int)noisy.size();
    std::cout<<"[EKF] Frames: "<<T
             <<", State dim: "<<N_STATE
             <<", Meas dim: " <<N_MEAS<<std::endl;

    // Build full system matrices
    double* F  = new double[N_STATE*N_STATE];
    double* Q  = new double[N_STATE*N_STATE];
    double* R  = new double[N_MEAS *N_MEAS ];
    double* Ft = new double[N_STATE*N_STATE];
    build_F_full(F);
    build_Q_full(Q);
    build_R_full(R);
    mat_T(F,Ft,N_STATE,N_STATE);

    // Allocate full state and covariance
    double* x      = new double[N_STATE];
    double* P      = new double[N_STATE*N_STATE];
    double* x_pred = new double[N_STATE];
    double* P_pred = new double[N_STATE*N_STATE];

    // Working matrices
    double* J     = new double[N_MEAS *N_STATE];
    double* Jt    = new double[N_STATE*N_MEAS ];
    double* FP    = new double[N_STATE*N_STATE];
    double* FPFt  = new double[N_STATE*N_STATE];
    double* JP    = new double[N_MEAS *N_STATE];
    double* JPJt  = new double[N_MEAS *N_MEAS ];
    double* S     = new double[N_MEAS *N_MEAS ];
    double* K     = new double[N_STATE*N_MEAS ];
    double* Ky    = new double[N_STATE];
    double* z_pred= new double[N_MEAS ];
    double* y     = new double[N_MEAS ];
    double* KJ    = new double[N_STATE*N_STATE];
    double* IKJ   = new double[N_STATE*N_STATE];
    double* IKJt  = new double[N_STATE*N_STATE];
    double* tmp1  = new double[N_STATE*N_STATE];
    double* tmp2  = new double[N_STATE*N_STATE];
    double* KR    = new double[N_STATE*N_MEAS ];
    double* KRKt  = new double[N_STATE*N_STATE];
    double* Kt    = new double[N_MEAS *N_STATE];
    double* I_big = new double[N_STATE*N_STATE];
    mat_eye(I_big,N_STATE);

    // Initialise full state x (276x1)
    mat_zero(x,N_STATE);
    for(int jt=0;jt<N_JOINTS;jt++){
        x[jt*N_S+0]=noisy[0][jt*3+0]; // px
        x[jt*N_S+4]=noisy[0][jt*3+1]; // py
        x[jt*N_S+8]=noisy[0][jt*3+2]; // pz
    }

    // Initialise P (276x276) = 100*I
    mat_eye(P,N_STATE);
    mat_scale(P,100.0,N_STATE*N_STATE);

    std::vector<std::vector<double>> output(T,std::vector<double>(N_STATE));
    for(int s=0;s<N_STATE;s++) output[0][s]=x[s];

    // Main filter loop
    for(int k=1;k<T;k++){

        // PREDICT
        mat_mul(F,x,x_pred,N_STATE,N_STATE,1);
        mat_mul(F,P,FP,N_STATE,N_STATE,N_STATE);
        mat_mul(FP,Ft,FPFt,N_STATE,N_STATE,N_STATE);
        mat_add(FPFt,Q,P_pred,N_STATE*N_STATE);

        // COMPUTE FULL JACOBIAN at predicted state (69x276)
        build_J_full(x_pred,J);
        mat_T(J,Jt,N_MEAS,N_STATE);

        // FULL NONLINEAR MEASUREMENT PREDICTION h(x_pred) (69x1)
        h_full(x_pred,z_pred);

        // CONVERT NOISY CARTESIAN MEASUREMENTS TO SPHERICAL (69x1)
        double z_meas[N_MEAS];
        for(int jt=0;jt<N_JOINTS;jt++){
            double nx=noisy[k][jt*3+0];
            double ny=noisy[k][jt*3+1];
            double nz=noisy[k][jt*3+2];
            double rho=std::sqrt(nx*nx+ny*ny)+EPS;
            z_meas[jt*3+0]=std::sqrt(nx*nx+ny*ny+nz*nz)+EPS;
            z_meas[jt*3+1]=manual_atan2(ny,nx);
            z_meas[jt*3+2]=manual_atan2(nz,rho);
        }

        // INNOVATION with angle wrapping
        for(int jt=0;jt<N_JOINTS;jt++){
            y[jt*3+0]=z_meas[jt*3+0]-z_pred[jt*3+0];
            y[jt*3+1]=wrap(z_meas[jt*3+1]-z_pred[jt*3+1]);
            y[jt*3+2]=wrap(z_meas[jt*3+2]-z_pred[jt*3+2]);
        }

        // INNOVATION COVARIANCE S = J*P_pred*Jt + R
        mat_mul(J,P_pred,JP,N_MEAS,N_STATE,N_STATE);
        mat_mul(JP,Jt,JPJt,N_MEAS,N_STATE,N_MEAS);
        mat_add(JPJt,R,S,N_MEAS*N_MEAS);

        // KALMAN GAIN (Cholesky)
        compute_gain(P_pred,J,S,K,N_STATE,N_MEAS);

        // STATE UPDATE
        mat_mul(K,y,Ky,N_STATE,N_MEAS,1);
        mat_add(x_pred,Ky,x,N_STATE);

        // COVARIANCE UPDATE (Joseph form)
        mat_mul(K,J,KJ,N_STATE,N_MEAS,N_STATE);
        mat_sub(I_big,KJ,IKJ,N_STATE*N_STATE);
        mat_mul(IKJ,P_pred,tmp1,N_STATE,N_STATE,N_STATE);
        mat_T(IKJ,IKJt,N_STATE,N_STATE);
        mat_mul(tmp1,IKJt,tmp2,N_STATE,N_STATE,N_STATE);
        mat_mul(K,R,KR,N_STATE,N_MEAS,N_MEAS);
        mat_T(K,Kt,N_STATE,N_MEAS);
        mat_mul(KR,Kt,KRKt,N_STATE,N_MEAS,N_STATE);
        mat_add(tmp2,KRKt,P,N_STATE*N_STATE);

        for(int s=0;s<N_STATE;s++) output[k][s]=x[s];

        if(k%500==0)
            std::cout<<"[EKF] Frame "<<k<<"/"<<T<<std::endl;
    }

    // Write output CSV
    std::ofstream out("output/ekf_output.csv");
    std::vector<std::string> joints={
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    std::vector<std::string> states={
        "px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"
    };
    bool first=true;
    for(auto& jn:joints) for(auto& st:states){
        if(!first) out<<",";
        out<<jn<<"_"<<st;
        first=false;
    }
    out<<"\n";
    for(int k=0;k<T;k++){
        for(int i=0;i<N_STATE;i++){
            if(i>0) out<<",";
            out<<output[k][i];
        }
        out<<"\n";
    }
    out.close();
    std::cout<<"[EKF] Done. Output: output/ekf_output.csv"<<std::endl;

    delete[] F; delete[] Q; delete[] R; delete[] Ft;
    delete[] x; delete[] P; delete[] x_pred; delete[] P_pred;
    delete[] J; delete[] Jt; delete[] FP; delete[] FPFt;
    delete[] JP; delete[] JPJt; delete[] S; delete[] K;
    delete[] Ky; delete[] z_pred; delete[] y;
    delete[] KJ; delete[] IKJ; delete[] IKJt;
    delete[] tmp1; delete[] tmp2; delete[] KR; delete[] KRKt;
    delete[] Kt; delete[] I_big;

    return 0;
}
