# =============================================================================
#  lkf_vector.s  —  Linear Kalman Filter: RISC-V Vectorised Assembly (RVV)
#
#  Milestone 4 — Member 1
#
#  This file provides vectorised drop-in replacements for the scalar
#  kernels in lkf_asm.s.  The C driver (lkf_main.c) calls the same
#  function names, so no changes to the C side are required.
#
#  Functions implemented (vectorised with RVV):
#    1.  mat_zero        — vle64 / vse64 zero-fill
#    2.  mat_eye         — vectorised zero + scalar diagonal
#    3.  mat_mul         — row-dot-product with vfmul/vfmacc
#    4.  mat_add         — element-wise vfadd
#    5.  mat_sub         — element-wise vfsub
#    6.  mat_transpose   — scalar (structure-dependent, not vectorisable simply)
#    7.  mat_scale       — vfmul by broadcast scalar
#    8.  chol_solve      — scalar (3×3, tiny — vectorising adds no benefit)
#    9.  compute_gain    — vectorised inner loop
#   10.  lkf_predict     — calls vectorised mat_mul / mat_add
#   11.  lkf_update      — calls vectorised kernels throughout
#
#  RVV conventions used:
#    • vsetvli  t0, aN, e64, m1, ta, ma   — set VL for double (e64)
#    • vle64.v  vD, (ptr)                  — unit-stride load
#    • vse64.v  vD, (ptr)                  — unit-stride store
#    • vfmul.vv, vfadd.vv, vfsub.vv       — element-wise FP ops
#    • vfmacc.vv vD, vA, vB               — vD += vA * vB  (fused MAC)
#    • vfmv.v.f vD, fA                    — broadcast scalar → vector
#    • vfredosum.vs vD, vS, vZ            — ordered reduction sum
#
#  Memory layout: row-major doubles, 8 bytes per element.
#  Pointer alignment: all matrices passed from lkf_main.c are malloc'd
#  (8-byte aligned). The vectorised loads/stores use unit stride and
#  require only natural alignment (8 bytes for e64), which is guaranteed.
#
#  Calling convention: RISC-V LP64D (same as M3 scalar)
#    Integer args  : a0–a7
#    FP args       : fa0–fa7
#    Saved regs    : s0–s11, fs0–fs11  (must be preserved across calls)
#    Temporaries   : t0–t6, ft0–ft11   (caller-saved)
#    Return addr   : ra
# =============================================================================

    .section .text

# =============================================================================
# 1. mat_zero(double* A, int r, int c)
#    Vectorised: fill r*c doubles with 0.0 using vse64.v
#
#    Args:
#      a0 = double* A
#      a1 = int r
#      a2 = int c
# =============================================================================
    .global mat_zero
mat_zero:
    mul     a3, a1, a2              # a3 = total elements = r*c
    beqz    a3, mat_zero_ret

    # Use RVV to zero elements in chunks of VL
    # vmv.v.i sets integer vector — we broadcast 0.0 via vfmv.v.f
    fmv.d.x ft0, zero               # ft0 = 0.0

mat_zero_vec_loop:
    beqz    a3, mat_zero_ret
    vsetvli t0, a3, e64, m8, ta, ma # t0 = VL (use m8 for max throughput)
    vfmv.v.f v0, ft0                # v0[0..VL-1] = 0.0
    vse64.v  v0, (a0)               # store VL doubles to A
    slli    t1, t0, 3               # t1 = VL * 8 bytes
    add     a0, a0, t1              # advance pointer
    sub     a3, a3, t0              # remaining elements
    j       mat_zero_vec_loop

mat_zero_ret:
    ret


# =============================================================================
# 2. mat_eye(double* A, int n)
#    Zero the matrix (vectorised), then set diagonal (scalar loop).
#
#    Args:
#      a0 = double* A
#      a1 = int n
# =============================================================================
    .global mat_eye
mat_eye:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0, 16(sp)
    sd      s1,  8(sp)

    mv      s0, a0                  # save base
    mv      s1, a1                  # save n

    # zero entire matrix using vectorised mat_zero
    mv      a2, a1                  # c = n
    call    mat_zero

    # set diagonal: A[i*(n+1)] = 1.0  (scalar — one write per diagonal)
    li      t0, 1
    fcvt.d.w ft1, t0                # ft1 = 1.0
    mv      a0, s0
    mv      a4, s1                  # loop counter = n
    addi    t1, s1, 1               # stride = n+1 elements
    slli    t1, t1, 3               # stride in bytes

mat_eye_diag:
    beqz    a4, mat_eye_done
    fsd     ft1, 0(a0)
    add     a0, a0, t1
    addi    a4, a4, -1
    j       mat_eye_diag

mat_eye_done:
    ld      ra, 24(sp)
    ld      s0, 16(sp)
    ld      s1,  8(sp)
    addi    sp, sp, 32
    ret


# =============================================================================
# 3. mat_mul(double* A, double* B, double* C, int r, int k, int c)
#    C = A * B   where A is r×k, B is k×c, C is r×c
#
#    Vectorisation strategy:
#      For each output row i and output column j:
#        C[i][j] = dot(A_row_i, B_col_j)   over k elements
#      We vectorise the dot product inner loop over k using vfmul+vfredosum.
#      The outer i,j loops remain scalar (typical k values are 12 or 276).
#
#    Args:
#      a0 = double* A
#      a1 = double* B
#      a2 = double* C
#      a3 = int r
#      a4 = int k
#      a5 = int c
#
#    Register allocation:
#      s0–s5  : A, B, C, r, k, c
#      s6     : row i
#      s7     : col j
#      s8     : A row base pointer (A + i*k*8)
#      s9     : scratch
#      ft0    : 0.0 for reduction init
# =============================================================================
    .global mat_mul
mat_mul:
    addi    sp, sp, -80
    sd      ra, 72(sp)
    sd      s0, 64(sp)
    sd      s1, 56(sp)
    sd      s2, 48(sp)
    sd      s3, 40(sp)
    sd      s4, 32(sp)
    sd      s5, 24(sp)
    sd      s6, 16(sp)
    sd      s7,  8(sp)
    sd      s8,  0(sp)

    mv      s0, a0                  # A
    mv      s1, a1                  # B
    mv      s2, a2                  # C
    mv      s3, a3                  # r
    mv      s4, a4                  # k
    mv      s5, a5                  # c

    fmv.d.x ft0, zero               # ft0 = 0.0 (for reduction init)

    li      s6, 0                   # i = 0
mat_mul_row:
    bge     s6, s3, mat_mul_done

    # A_row_base = A + i*k*8
    mul     t0, s6, s4              # t0 = i*k
    slli    t0, t0, 3               # bytes
    add     s8, s0, t0              # s8 = &A[i][0]

    li      s7, 0                   # j = 0
mat_mul_col:
    bge     s7, s5, mat_mul_col_done

    # Build B column j into a contiguous scratch is expensive.
    # Instead: load A row and B column simultaneously using strided loads.
    #
    # A row i is contiguous: stride = 8 bytes  → unit-stride vle64
    # B col j is strided:    stride = c*8 bytes → vlse64
    #
    # Compute dot product: sum = vfmul.vv(A_row, B_col) then vfredosum

    # B column j starts at B + j*8, stride = c*8
    slli    t1, s7, 3               # t1 = j*8  (byte offset to col j)
    add     t2, s1, t1              # t2 = &B[0][j]
    slli    t3, s5, 3               # t3 = c*8  (column stride in bytes)

    # We process k elements in chunks of VL
    mv      t4, s4                  # remaining = k
    mv      t5, s8                  # ptr into A row
    mv      t6, t2                  # ptr into B column

    # initialise scalar accumulator = 0.0
    fmv.d.x fa0, zero

mat_mul_dot:
    beqz    t4, mat_mul_dot_done
    vsetvli a6, t4, e64, m4, ta, ma # VL = min(t4, VLMAX/4)

    # Load VL elements of A row (unit stride)
    vle64.v  v0, (t5)
    slli    a7, a6, 3
    add     t5, t5, a7              # advance A row pointer

    # Load VL elements of B column (strided)
    vlse64.v v4, (t6), t3
    # advance B col pointer by VL*stride
    mul     a7, a6, t3
    add     t6, t6, a7

    # Element-wise multiply: v8 = v0 * v4
    vfmul.vv v8, v0, v4

    # Ordered reduction sum: v12[0] = sum(v8) + 0.0
    vsetvli zero, a6, e64, m4, ta, ma
    vfmv.s.f v12, ft0               # init reduction accumulator = 0.0
    vfredosum.vs v12, v8, v12       # v12[0] += sum(v8)

    # Extract scalar result and accumulate
    vfmv.f.s fa1, v12               # fa1 = v12[0]
    fadd.d  fa0, fa0, fa1           # accumulator += partial sum

    sub     t4, t4, a6
    j       mat_mul_dot

mat_mul_dot_done:
    # C[i][j] = fa0
    mul     t0, s6, s5              # t0 = i*c
    add     t0, t0, s7              # t0 = i*c + j
    slli    t0, t0, 3
    add     t0, s2, t0
    fsd     fa0, 0(t0)

    addi    s7, s7, 1
    j       mat_mul_col

mat_mul_col_done:
    addi    s6, s6, 1
    j       mat_mul_row

mat_mul_done:
    ld      ra, 72(sp)
    ld      s0, 64(sp)
    ld      s1, 56(sp)
    ld      s2, 48(sp)
    ld      s3, 40(sp)
    ld      s4, 32(sp)
    ld      s5, 24(sp)
    ld      s6, 16(sp)
    ld      s7,  8(sp)
    ld      s8,  0(sp)
    addi    sp, sp, 80
    ret


# =============================================================================
# 4. mat_add(double* A, double* B, double* C, int r, int c)
#    C = A + B  element-wise, vectorised with vfadd.vv
#
#    Args:
#      a0 = double* A
#      a1 = double* B
#      a2 = double* C
#      a3 = int r
#      a4 = int c
# =============================================================================
    .global mat_add
mat_add:
    mul     a3, a3, a4              # total elements
    beqz    a3, mat_add_ret

mat_add_vec_loop:
    beqz    a3, mat_add_ret
    vsetvli t0, a3, e64, m8, ta, ma
    vle64.v  v0, (a0)
    vle64.v  v8, (a1)
    vfadd.vv v16, v0, v8
    vse64.v  v16, (a2)
    slli    t1, t0, 3
    add     a0, a0, t1
    add     a1, a1, t1
    add     a2, a2, t1
    sub     a3, a3, t0
    j       mat_add_vec_loop

mat_add_ret:
    ret


# =============================================================================
# 5. mat_sub(double* A, double* B, double* C, int r, int c)
#    C = A - B  element-wise, vectorised with vfsub.vv
#
#    Args:
#      a0 = double* A
#      a1 = double* B
#      a2 = double* C
#      a3 = int r
#      a4 = int c
# =============================================================================
    .global mat_sub
mat_sub:
    mul     a3, a3, a4
    beqz    a3, mat_sub_ret

mat_sub_vec_loop:
    beqz    a3, mat_sub_ret
    vsetvli t0, a3, e64, m8, ta, ma
    vle64.v  v0, (a0)
    vle64.v  v8, (a1)
    vfsub.vv v16, v0, v8
    vse64.v  v16, (a2)
    slli    t1, t0, 3
    add     a0, a0, t1
    add     a1, a1, t1
    add     a2, a2, t1
    sub     a3, a3, t0
    j       mat_sub_vec_loop

mat_sub_ret:
    ret


# =============================================================================
# 6. mat_transpose(double* A, double* B, int r, int c)
#    B = Aᵀ
#
#    Transposition is inherently gather/scatter.  For the small matrices
#    used here (12×12, 3×12, 12×3) the scalar implementation from M3
#    is retained — the strided gather overhead would exceed the benefit.
#    The function signature is identical to M3.
#
#    Args:
#      a0 = double* A  (r×c)
#      a1 = double* B  (c×r)
#      a2 = int r
#      a3 = int c
# =============================================================================
    .global mat_transpose
mat_transpose:
    # scalar nested loop: B[j*r+i] = A[i*c+j]
    mv      t4, a2                  # r
    mv      t5, a3                  # c
    li      t0, 0                   # i = 0
mat_trans_i:
    bge     t0, t4, mat_trans_done
    li      t1, 0                   # j = 0
mat_trans_j:
    bge     t1, t5, mat_trans_j_done
    # A[i*c+j]
    mul     t2, t0, t5
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t2, a0, t2
    fld     ft0, 0(t2)
    # B[j*r+i]
    mul     t3, t1, t4
    add     t3, t3, t0
    slli    t3, t3, 3
    add     t3, a1, t3
    fsd     ft0, 0(t3)
    addi    t1, t1, 1
    j       mat_trans_j
mat_trans_j_done:
    addi    t0, t0, 1
    j       mat_trans_i
mat_trans_done:
    ret


# =============================================================================
# 7. mat_scale(double* A, double s, double* B, int r, int c)
#    B = s * A  element-wise, vectorised with vfmul.vf (broadcast scalar)
#
#    Args:
#      a0 = double* A
#      fa0= double  s   (FP scalar in fa0)
#      a1 = double* B
#      a2 = int r
#      a3 = int c
# =============================================================================
    .global mat_scale
mat_scale:
    mul     a2, a2, a3              # total elements
    beqz    a2, mat_scale_ret

mat_scale_vec_loop:
    beqz    a2, mat_scale_ret
    vsetvli t0, a2, e64, m8, ta, ma
    vle64.v  v0, (a0)
    vfmul.vf v8, v0, fa0            # v8[i] = v0[i] * s
    vse64.v  v8, (a1)
    slli    t1, t0, 3
    add     a0, a0, t1
    add     a1, a1, t1
    sub     a2, a2, t0
    j       mat_scale_vec_loop

mat_scale_ret:
    ret


# =============================================================================
# 8. chol_solve(double* A, double* b, double* x, int n)
#    Solve A*x = b where A is n×n SPD via Cholesky.
#    n = 3 always in this filter (innovation covariance S is 3×3).
#    Scalar implementation retained — 3×3 Cholesky is 27 operations;
#    vectorising a 3-element problem adds setup overhead with no gain.
#
#    This is an exact copy of the M3 scalar implementation.
# =============================================================================
    .global chol_solve
chol_solve:
    addi    sp, sp, -128
    sd      ra, 120(sp)
    sd      s0, 112(sp)
    sd      s1, 104(sp)
    sd      s2,  96(sp)
    sd      s3,  88(sp)
    sd      s4,  80(sp)

    mv      s0, a0          # A
    mv      s1, a1          # b
    mv      s2, a2          # x
    mv      s3, a3          # n

    # allocate L on stack: n*n doubles = 9*8 = 72 bytes (n=3)
    addi    sp, sp, -72
    mv      s4, sp          # s4 = L base

    # zero L
    li      t0, 0
chol_lzero:
    bge     t0, s3, chol_lzero_done
    mv      t1, s3
    mul     t2, t0, s3
chol_lzero_j:
    bge     t1, s3, chol_lzero_j_done  # BUG: fix condition
    # L[i*n+j] = 0
    add     t3, t2, t0      # reuse
    j       chol_lzero_j
chol_lzero_j_done:
    addi    t0, t0, 1
    j       chol_lzero
chol_lzero_done:

    # --- simplified direct 3x3 Cholesky for n=3 ---
    # We hardcode n=3 for clarity and correctness.
    # L[0][0] = sqrt(A[0][0])
    fld     ft0,  0(s0)         # A[0][0]
    fsqrt.d ft0, ft0
    fsd     ft0,  0(s4)         # L[0][0]

    # L[1][0] = A[1][0] / L[0][0]
    fld     ft1,  24(s0)        # A[1][0]  (row 1, col 0: offset 3*8=24? no: (1*3+0)*8=8 wait n=3)
    # A[i*n+j]*8: A[1*3+0]=A[3] => offset 3*8=24
    fld     ft1,  24(s0)
    fdiv.d  ft1, ft1, ft0
    fsd     ft1,  24(s4)        # L[1][0]

    # L[1][1] = sqrt(A[1][1] - L[1][0]^2)
    fld     ft2,  32(s0)        # A[1][1] offset=(1*3+1)*8=32
    fmul.d  ft3, ft1, ft1
    fsub.d  ft2, ft2, ft3
    fsqrt.d ft2, ft2
    fsd     ft2,  32(s4)        # L[1][1]

    # L[2][0] = A[2][0] / L[0][0]
    fld     ft4,  48(s0)        # A[2][0] offset=(2*3+0)*8=48
    fdiv.d  ft4, ft4, ft0
    fsd     ft4,  48(s4)

    # L[2][1] = (A[2][1] - L[2][0]*L[1][0]) / L[1][1]
    fld     ft5,  56(s0)        # A[2][1] offset=(2*3+1)*8=56
    fmul.d  ft6, ft4, ft1
    fsub.d  ft5, ft5, ft6
    fdiv.d  ft5, ft5, ft2
    fsd     ft5,  56(s4)

    # L[2][2] = sqrt(A[2][2] - L[2][0]^2 - L[2][1]^2)
    fld     ft6,  64(s0)        # A[2][2] offset=(2*3+2)*8=64
    fmul.d  ft7, ft4, ft4
    fsub.d  ft6, ft6, ft7
    fmul.d  ft7, ft5, ft5
    fsub.d  ft6, ft6, ft7
    fsqrt.d ft6, ft6
    fsd     ft6,  64(s4)

    # Forward substitution: L*y = b  (y stored in x temporarily)
    # y[0] = b[0] / L[0][0]
    fld     ft0,   0(s1)
    fld     ft1,   0(s4)        # L[0][0]
    fdiv.d  ft0, ft0, ft1
    fsd     ft0,   0(s2)

    # y[1] = (b[1] - L[1][0]*y[0]) / L[1][1]
    fld     ft1,   8(s1)        # b[1]
    fld     ft2,  24(s4)        # L[1][0]
    fld     ft3,   0(s2)        # y[0]
    fmul.d  ft4, ft2, ft3
    fsub.d  ft1, ft1, ft4
    fld     ft2,  32(s4)        # L[1][1]
    fdiv.d  ft1, ft1, ft2
    fsd     ft1,   8(s2)

    # y[2] = (b[2] - L[2][0]*y[0] - L[2][1]*y[1]) / L[2][2]
    fld     ft1,  16(s1)        # b[2]
    fld     ft2,  48(s4)        # L[2][0]
    fld     ft3,   0(s2)        # y[0]
    fmul.d  ft4, ft2, ft3
    fsub.d  ft1, ft1, ft4
    fld     ft2,  56(s4)        # L[2][1]
    fld     ft3,   8(s2)        # y[1]
    fmul.d  ft4, ft2, ft3
    fsub.d  ft1, ft1, ft4
    fld     ft2,  64(s4)        # L[2][2]
    fdiv.d  ft1, ft1, ft2
    fsd     ft1,  16(s2)

    # Backward substitution: Lᵀ*x = y
    # x[2] = y[2] / L[2][2]
    fld     ft0,  16(s2)
    fld     ft1,  64(s4)
    fdiv.d  ft0, ft0, ft1
    fsd     ft0,  16(s2)

    # x[1] = (y[1] - L[2][1]*x[2]) / L[1][1]
    fld     ft1,   8(s2)
    fld     ft2,  56(s4)        # L[2][1] = Lᵀ[1][2]
    fmul.d  ft3, ft2, ft0
    fsub.d  ft1, ft1, ft3
    fld     ft2,  32(s4)
    fdiv.d  ft1, ft1, ft2
    fsd     ft1,   8(s2)

    # x[0] = (y[0] - L[1][0]*x[1] - L[2][0]*x[2]) / L[0][0]
    fld     ft0,   0(s2)
    fld     ft2,  24(s4)        # L[1][0]
    fld     ft3,   8(s2)        # x[1]
    fmul.d  ft4, ft2, ft3
    fsub.d  ft0, ft0, ft4
    fld     ft2,  48(s4)        # L[2][0]
    fld     ft3,  16(s2)        # x[2]
    fmul.d  ft4, ft2, ft3
    fsub.d  ft0, ft0, ft4
    fld     ft2,   0(s4)        # L[0][0]
    fdiv.d  ft0, ft0, ft2
    fsd     ft0,   0(s2)

    addi    sp, sp, 72          # free L stack space
    ld      ra, 120(sp)
    ld      s0, 112(sp)
    ld      s1, 104(sp)
    ld      s2,  96(sp)
    ld      s3,  88(sp)
    ld      s4,  80(sp)
    addi    sp, sp, 128
    ret


# =============================================================================
# 9. compute_gain(double* P, double* H, double* S, double* K)
#    K = P * Hᵀ * S⁻¹
#
#    For each column j of K (j = 0,1,2):
#      1. Solve S * e_j → Sinv_col  (3×1)
#      2. K[:,j] = (P * Hᵀ) * Sinv_col  (12×1)
#         This is a mat_vec product of PHt (12×3) with Sinv_col (3×1).
#         We vectorise step 2 using vfmacc across the 3-element inner sum.
#
#    Function uses the same signature as M3:
#      a0 = P  (12×12)
#      a1 = H  (3×12)
#      a2 = S  (3×3)
#      a3 = K  (12×3)
#
#    This function is called from C with fixed dimensions so we use them directly.
# =============================================================================
    .global compute_gain
compute_gain:
    addi    sp, sp, -128
    sd      ra, 120(sp)
    sd      s0, 112(sp)
    sd      s1, 104(sp)
    sd      s2,  96(sp)
    sd      s3,  88(sp)
    sd      s4,  80(sp)
    sd      s5,  72(sp)
    sd      s6,  64(sp)

    mv      s0, a0              # P  (12×12)
    mv      s1, a1              # H  (3×12)
    mv      s2, a2              # S  (3×3)
    mv      s3, a3              # K  (12×3)

    # Allocate PHt (12×3 = 36 doubles = 288 bytes) and Ht (12×3=288 bytes) on stack
    addi    sp, sp, -576
    mv      s4, sp              # s4 = PHt base  (12×3)
    addi    s5, sp, 288         # s5 = Ht base   (12×3)

    # Compute Ht = Hᵀ  (H is 3×12, Ht is 12×3)
    mv      a0, s1              # H
    mv      a1, s5              # Ht
    li      a2, 3               # r = 3
    li      a3, 12              # c = 12
    call    mat_transpose

    # Compute PHt = P * Ht  (12×12 * 12×3 = 12×3)
    mv      a0, s0              # P
    mv      a1, s5              # Ht
    mv      a2, s4              # PHt
    li      a3, 12
    li      a4, 12
    li      a5, 3
    call    mat_mul

    # For each column j = 0,1,2 of K:
    li      s6, 0               # j = 0

compute_gain_col:
    li      t0, 3
    bge     s6, t0, compute_gain_done

    # Step 1: solve S * Sinv_col = e_j
    # Allocate e_j (3 doubles) and Sinv_col (3 doubles) on stack
    addi    sp, sp, -48
    # e_j at sp, Sinv_col at sp+24
    fmv.d.x ft0, zero
    fsd     ft0,  0(sp)
    fsd     ft0,  8(sp)
    fsd     ft0, 16(sp)         # e_j = [0,0,0]
    li      t1, 1
    fcvt.d.w ft1, t1            # ft1 = 1.0
    slli    t2, s6, 3           # offset = j*8
    add     t3, sp, t2
    fsd     ft1, 0(t3)          # e_j[j] = 1.0

    mv      a0, s2              # S
    mv      a1, sp              # e_j
    addi    a2, sp, 24          # Sinv_col
    li      a3, 3
    call    chol_solve

    # Step 2: K[:,j] = PHt (12×3) * Sinv_col (3×1) using vectorised dot
    # For each row r of K (r = 0..11):
    #   K[r][j] = PHt[r][0]*Sinv_col[0] + PHt[r][1]*Sinv_col[1] + PHt[r][2]*Sinv_col[2]
    #
    # Load Sinv_col into scalar FP regs
    addi    t0, sp, 24          # Sinv_col base
    fld     fs0, 0(t0)          # Sinv_col[0]
    fld     fs1, 8(t0)          # Sinv_col[1]
    fld     fs2, 16(t0)         # Sinv_col[2]

    # Broadcast each Sinv_col element into vector for vfmacc
    li      t1, 12              # 12 rows in PHt
    mv      t2, s4              # PHt base pointer (row 0, col 0)
    # K[:,j] base: K + j*8, stride = 3*8 = 24 bytes per row
    slli    t3, s6, 3           # j*8
    add     t4, s3, t3          # K + j*8 = &K[0][j]
    li      t5, 24              # stride = 3 doubles * 8 bytes

    # Process all 12 rows at once using vector strided store
    # PHt row r starts at s4 + r*3*8
    # Load all 12 elements of PHt column 0 (strided: stride=3*8=24)
    li      a3, 12
    vsetvli zero, a3, e64, m4, ta, ma

    # PHt col 0: base=s4+0, stride=24
    li      t6, 24
    vlse64.v v0, (s4), t6       # v0 = PHt[:,0]

    # PHt col 1: base=s4+8, stride=24
    addi    a4, s4, 8
    vlse64.v v4, (a4), t6       # v4 = PHt[:,1]

    # PHt col 2: base=s4+16, stride=24
    addi    a4, s4, 16
    vlse64.v v8, (a4), t6       # v8 = PHt[:,2]

    # result = PHt[:,0]*Sinv_col[0] + PHt[:,1]*Sinv_col[1] + PHt[:,2]*Sinv_col[2]
    vfmul.vf  v12, v0, fs0      # v12 = PHt[:,0] * Sinv_col[0]
    vfmacc.vf v12, fs1, v4      # v12 += PHt[:,1] * Sinv_col[1]
    vfmacc.vf v12, fs2, v8      # v12 += PHt[:,2] * Sinv_col[2]

    # Store K[:,j] with stride 24 (= 3 doubles * 8 bytes per row)
    vsse64.v v12, (t4), t5

    addi    sp, sp, 48          # free e_j + Sinv_col
    addi    s6, s6, 1
    j       compute_gain_col

compute_gain_done:
    addi    sp, sp, 576         # free PHt + Ht
    ld      ra, 120(sp)
    ld      s0, 112(sp)
    ld      s1, 104(sp)
    ld      s2,  96(sp)
    ld      s3,  88(sp)
    ld      s4,  80(sp)
    ld      s5,  72(sp)
    ld      s6,  64(sp)
    addi    sp, sp, 128
    ret


# =============================================================================
# 10. lkf_predict(F, Ft, P, Q, x, x_pred, P_pred, FP, FPFt)
#     Predict step:
#       x_pred = F * x          (12×12 * 12×1)
#       FP     = F * P          (12×12 * 12×12)
#       FPFt   = FP * Ft        (12×12 * 12×12)
#       P_pred = FPFt + Q       (12×12 + 12×12)
#
#     All sub-operations delegated to vectorised mat_mul / mat_add above.
#     This function is identical in structure to M3 — the speedup comes
#     from the vectorised kernels it calls.
#
#     Args (a0–a7 = first 8, FPFt on stack):
#       a0 = F      (12×12)
#       a1 = Ft     (12×12)
#       a2 = P      (12×12)
#       a3 = Q      (12×12)
#       a4 = x      (12×1)
#       a5 = x_pred (12×1)
#       a6 = P_pred (12×12)
#       a7 = FP     (12×12)
#     Stack +0: FPFt (12×12)
# =============================================================================
    .global lkf_predict
lkf_predict:
    addi    sp, sp, -96
    sd      ra,  88(sp)
    sd      s0,  80(sp)
    sd      s1,  72(sp)
    sd      s2,  64(sp)
    sd      s3,  56(sp)
    sd      s4,  48(sp)
    sd      s5,  40(sp)
    sd      s6,  32(sp)
    sd      s7,  24(sp)
    sd      s8,  16(sp)

    mv      s0, a0              # F
    mv      s1, a1              # Ft
    mv      s2, a2              # P
    mv      s3, a3              # Q
    mv      s4, a4              # x
    mv      s5, a5              # x_pred
    mv      s6, a6              # P_pred
    mv      s7, a7              # FP
    ld      s8, 96(sp)          # FPFt (from caller stack, offset = frame size)

    # x_pred = F * x  (12×12 * 12×1 = 12×1)
    mv      a0, s0
    mv      a1, s4
    mv      a2, s5
    li      a3, 12
    li      a4, 12
    li      a5, 1
    call    mat_mul

    # FP = F * P  (12×12 * 12×12 = 12×12)
    mv      a0, s0
    mv      a1, s2
    mv      a2, s7
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # FPFt = FP * Ft  (12×12 * 12×12 = 12×12)
    mv      a0, s7
    mv      a1, s1
    mv      a2, s8
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # P_pred = FPFt + Q
    mv      a0, s8
    mv      a1, s3
    mv      a2, s6
    li      a3, 12
    li      a4, 12
    call    mat_add

    ld      ra,  88(sp)
    ld      s0,  80(sp)
    ld      s1,  72(sp)
    ld      s2,  64(sp)
    ld      s3,  56(sp)
    ld      s4,  48(sp)
    ld      s5,  40(sp)
    ld      s6,  32(sp)
    ld      s7,  24(sp)
    ld      s8,  16(sp)
    addi    sp, sp, 96
    ret


# =============================================================================
# 11. lkf_update(K, H, P_pred, R, x_pred, y, x, P,
#                KH, IKH, tmp1, tmp2, KR, KRKt, Kt, I12)
#     Joseph-form covariance update — all sub-operations vectorised.
#
#     Args (a0–a7 = first 8, rest on stack at old_sp+0..+56):
#       a0 = K      (12×3)       a1 = H  (3×12)
#       a2 = P_pred (12×12)      a3 = R  (3×3)
#       a4 = x_pred (12×1)       a5 = y  (3×1)
#       a6 = x      (12×1) out   a7 = P  (12×12) out
#     Stack:
#       +0  KH(12×12)  +8  IKH(12×12)  +16 tmp1(12×12)  +24 tmp2(12×12)
#       +32 KR(12×3)   +40 KRKt(12×12) +48 Kt(3×12)     +56 I12(12×12)
# =============================================================================
    .global lkf_update
lkf_update:
    addi    sp, sp, -112
    sd      ra,  104(sp)
    sd      s0,   96(sp)
    sd      s1,   88(sp)
    sd      s2,   80(sp)
    sd      s3,   72(sp)
    sd      s4,   64(sp)
    sd      s5,   56(sp)
    sd      s6,   48(sp)
    sd      s7,   40(sp)
    sd      s8,   32(sp)
    sd      s9,   24(sp)
    sd      s10,  16(sp)
    sd      s11,   8(sp)

    mv      s0,  a0             # K
    mv      s1,  a1             # H
    mv      s2,  a2             # P_pred
    mv      s3,  a3             # R
    mv      s4,  a4             # x_pred
    mv      s5,  a5             # y
    mv      s6,  a6             # x (output)
    mv      s7,  a7             # P (output)
    ld      s8,  112(sp)        # KH
    ld      s9,  120(sp)        # IKH
    ld      s10, 128(sp)        # tmp1
    ld      s11, 136(sp)        # tmp2
    addi    sp, sp, -32
    ld      t3,  144+32(sp)     # KR    — adjust for inner frame
    ld      t4,  152+32(sp)     # KRKt
    ld      t5,  160+32(sp)     # Kt
    ld      t6,  168+32(sp)     # I12
    sd      t3,  24(sp)
    sd      t4,  16(sp)
    sd      t5,   8(sp)
    sd      t6,   0(sp)

    # ── x = x_pred + K*y ─────────────────────────────────────────────────
    mv      a0, s0              # K (12×3)
    mv      a1, s5              # y (3×1)
    mv      a2, s6              # x (12×1) — store K*y here
    li      a3, 12
    li      a4, 3
    li      a5, 1
    call    mat_mul             # x = K*y

    mv      a0, s4              # x_pred
    mv      a1, s6              # K*y
    mv      a2, s6              # x = x_pred + K*y
    li      a3, 12
    li      a4, 1
    call    mat_add

    # ── KH = K * H  (12×3 * 3×12 = 12×12) ───────────────────────────────
    mv      a0, s0
    mv      a1, s1
    mv      a2, s8
    li      a3, 12
    li      a4, 3
    li      a5, 12
    call    mat_mul

    # ── IKH = I12 - KH ───────────────────────────────────────────────────
    ld      t6,  0(sp)          # I12
    mv      a0, t6
    mv      a1, s8
    mv      a2, s9
    li      a3, 12
    li      a4, 12
    call    mat_sub

    # ── tmp1 = IKH * P_pred ──────────────────────────────────────────────
    mv      a0, s9
    mv      a1, s2
    mv      a2, s10
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # ── IKHt = IKHᵀ (store in KRKt buffer temporarily) ──────────────────
    ld      t4,  16(sp)         # KRKt buffer (reused as IKHt scratch)
    mv      a0, s9              # IKH
    mv      a1, t4
    li      a2, 12
    li      a3, 12
    call    mat_transpose

    # ── tmp2 = tmp1 * IKHt ───────────────────────────────────────────────
    mv      a0, s10
    mv      a1, t4
    mv      a2, s11
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # ── KR = K * R  (12×3 * 3×3 = 12×3) ─────────────────────────────────
    ld      t3,  24(sp)         # KR
    mv      a0, s0
    mv      a1, s3
    mv      a2, t3
    li      a3, 12
    li      a4, 3
    li      a5, 3
    call    mat_mul

    # ── Kt = Kᵀ  (12×3 → 3×12) ───────────────────────────────────────────
    ld      t5,   8(sp)         # Kt
    mv      a0, s0
    mv      a1, t5
    li      a2, 12
    li      a3, 3
    call    mat_transpose

    # ── KRKt = KR * Kt  (12×3 * 3×12 = 12×12) ───────────────────────────
    ld      t4,  16(sp)         # KRKt (real destination now)
    mv      a0, t3
    mv      a1, t5
    mv      a2, t4
    li      a3, 12
    li      a4, 3
    li      a5, 12
    call    mat_mul

    # ── P = tmp2 + KRKt ──────────────────────────────────────────────────
    mv      a0, s11
    mv      a1, t4
    mv      a2, s7
    li      a3, 12
    li      a4, 12
    call    mat_add

    addi    sp, sp, 32          # pop inner frame
    ld      ra,  104(sp)
    ld      s0,   96(sp)
    ld      s1,   88(sp)
    ld      s2,   80(sp)
    ld      s3,   72(sp)
    ld      s4,   64(sp)
    ld      s5,   56(sp)
    ld      s6,   48(sp)
    ld      s7,   40(sp)
    ld      s8,   32(sp)
    ld      s9,   24(sp)
    ld      s10,  16(sp)
    ld      s11,   8(sp)
    addi    sp, sp, 112
    ret

# end of lkf_vector.s
