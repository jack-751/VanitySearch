/*
 * CalcR_GPU.cu
 *
 * CPU 產生隨機 k 值 → GPU 計算 r = x(k*G) mod n (secp256k1 ECDSA)
 *
 * 策略：
 *   ① 每個 CUDA thread 負責一個 k 值。
 *   ② 純量乘法採用「雙倍加法 (double-and-add)」256-bit 二進位展開。
 *   ③ 所有模數運算直接複用 VanitySearch 的 GPUMath.h 核心
 *      (_ModMult, _ModSqr, _ModInv) — 已是目前最快的 PTX 實作。
 *   ④ 當批次夠大時，可改用 Montgomery Batch Inverse 進一步提速，
 *      但 double-and-add 本身每步需要不同的 inv，所以這裡的最優策略
 *      是 Jacobian 座標（避免每步求逆），最後統一轉回仿射座標時做一次
 *      Batch Inverse。本程式以此方式實作。
 *
 * Jacobian 座標 (X:Y:Z) 對應仿射 (x, y) = (X/Z², Y/Z³)
 *   點加法 / 點倍增全程無需 ModInv。
 *   最後 Z != 0 時，x_affine = X * Z^{-2} mod P。
 *   最多只需 (n_keys/BATCH) 次 Batch Inverse，效率極高。
 *
 * 編譯 (在 /workspace 目錄下)：
 *   nvcc -O2 -arch=sm_86 -I. CalcR_GPU.cu -o CalcR_GPU -lhiredis $(pkg-config --cflags --libs libmongoc-1.0)
 *   (sm_86 = RTX 3xxx/Ada；依卡型調整)
 *
 * 執行：
 *   ./CalcR_GPU [count]  # count: 每批要計算的 k 值數量，預設 1048576
 *                        # 傳入 0 表示無限迴圈模式
 *                        # k 值由 CPU 透過 /dev/urandom 隨機產生
 */

 
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <unordered_set>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <hiredis/hiredis.h>
#include <mongoc/mongoc.h>
#include <bson/bson.h>
#include <ctime>
#include <cmath>
#include <chrono>

static void checkCuda(cudaError_t err, const char *msg);

// ===========================================================================
// 引入 VanitySearch 的 GPU 數學庫
// ===========================================================================
// 需要定義 NBBLOCK、IDX、GRP_SIZE (GPUMath.h 內部使用)
#define NBBLOCK  5
#define IDX      threadIdx.x
#define GRP_SIZE 1024          // GPUMath.h 中 _ModInvGrouped 需要此定義
#include "GPUMath.h"

// ===========================================================================
// secp256k1 常數 (host side)
// ===========================================================================
// P = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
static const uint64_t HOST_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
static const uint64_t HOST_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};
// G.x
static const uint64_t HOST_GX[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
};
// G.y
static const uint64_t HOST_GY[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
};

// ===========================================================================
// GPU 端裝置常數
// ===========================================================================
__device__ __constant__ uint64_t DEV_P[4];
__device__ __constant__ uint64_t DEV_N[4];
__device__ __constant__ uint64_t DEV_GX[4];
__device__ __constant__ uint64_t DEV_GY[4];
// 預計算表: DEV_TABLE[j] = (j+1)·G, j=0..14 (即 1·G ~ 15·G 的仿射座標)
__device__ __constant__ uint64_t DEV_TABLE_X[15][4];
__device__ __constant__ uint64_t DEV_TABLE_Y[15][4];

// ===========================================================================
// Jacobian 座標下的 secp256k1 點運算 (GPU __device__)
// 所有坐標均 < P，使用 _ModMult/_ModSqr (來自 GPUMath.h)
// ===========================================================================

// 輔助：完整縮減至 [0, P)
__device__ __forceinline__ void FullReduceP(uint64_t r[4]) {
    // _ModMult 結果保證在 [0, 2P) 附近，只需簡單減法
    uint64_t d[4] = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    int ge = (r[3] > d[3]) ||
             (r[3] == d[3] && r[2] > d[2]) ||
             (r[3] == d[3] && r[2] == d[2] && r[1] > d[1]) ||
             (r[3] == d[3] && r[2] == d[2] && r[1] == d[1] && r[0] >= d[0]);
    if (ge) {
        ModSub256(r, r, d);
    }
}

// ModAdd: r = (a + b) mod P
// 利用 ModSub256(a, ModNeg256(b)) 實現： a - (P - b) = a + b - P (若借位則 +P 得 a+b)
__device__ __forceinline__ void ModAddP(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t b_neg[4];
    uint64_t bc[4] = {b[0], b[1], b[2], b[3]};
    ModNeg256(b_neg, bc);
    uint64_t ac[4] = {a[0], a[1], a[2], a[3]};
    ModSub256(r, ac, b_neg);
}

// ModDouble: r = 2*a mod P
__device__ __forceinline__ void ModDoubleP(uint64_t r[4], const uint64_t a[4]) {
    ModAddP(r, a, a);
}

// r = a * b mod P
__device__ __forceinline__ void ModMulP(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t tmp_a[5], tmp_b[5];
    tmp_a[0]=a[0]; tmp_a[1]=a[1]; tmp_a[2]=a[2]; tmp_a[3]=a[3]; tmp_a[4]=0;
    tmp_b[0]=b[0]; tmp_b[1]=b[1]; tmp_b[2]=b[2]; tmp_b[3]=b[3]; tmp_b[4]=0;
    uint64_t res[5];
    _ModMult(res, tmp_a, tmp_b);
    r[0]=res[0]; r[1]=res[1]; r[2]=res[2]; r[3]=res[3];
    FullReduceP(r);
}

// r = a^2 mod P
__device__ __forceinline__ void ModSqP(uint64_t r[4], const uint64_t a[4]) {
    uint64_t tmp_a[5];
    tmp_a[0]=a[0]; tmp_a[1]=a[1]; tmp_a[2]=a[2]; tmp_a[3]=a[3]; tmp_a[4]=0;
    uint64_t res[4];
    _ModSqr(res, tmp_a);
    r[0]=res[0]; r[1]=res[1]; r[2]=res[2]; r[3]=res[3];
    FullReduceP(r);
}

// ===========================================================================
// Jacobian 點倍增:  (X3:Y3:Z3) = 2*(X1:Y1:Z1)
// 公式 (a=0, secp256k1):
//   S  = 4*X1*Y1^2
//   M  = 3*X1^2
//   X3 = M^2 - 2*S
//   Y3 = M*(S - X3) - 8*Y1^4
//   Z3 = 2*Y1*Z1
// ===========================================================================
__device__ void JacDouble(
    uint64_t X3[4], uint64_t Y3[4], uint64_t Z3[4],
    const uint64_t X1[4], const uint64_t Y1[4], const uint64_t Z1[4])
{
    uint64_t Y1sq[4], S[4], M[4], M2[4], Y1_4[4], tmp[4];

    // Y1^2
    ModSqP(Y1sq, Y1);
    // S = 4*X1*Y1^2
    ModMulP(S, X1, Y1sq);           // S = X1*Y1^2
    ModDoubleP(S, S);                // S = 2*X1*Y1^2
    ModDoubleP(S, S);                // S = 4*X1*Y1^2

    // M = 3*X1^2
    ModSqP(M, X1);                   // M = X1^2
    ModDoubleP(tmp, M);              // tmp = 2*X1^2
    ModAddP(M, M, tmp);              // M = 3*X1^2

    // X3 = M^2 - 2*S
    ModSqP(M2, M);                   // M2 = M^2
    ModDoubleP(tmp, S);              // tmp = 2*S
    ModSub256(X3, M2, tmp);         // X3 = M^2 - 2S

    // Y1^4
    ModSqP(Y1_4, Y1sq);             // Y1_4 = Y1^4

    // Y3 = M*(S - X3) - 8*Y1^4
    uint64_t tempS[4] = {S[0], S[1], S[2], S[3]};
    ModSub256(tmp, tempS, X3);       // tmp = S - X3
    ModMulP(Y3, M, tmp);            // Y3 = M*(S-X3)
    ModDoubleP(Y1_4, Y1_4);         // Y1_4 = 2*Y1^4
    ModDoubleP(Y1_4, Y1_4);         // Y1_4 = 4*Y1^4
    ModDoubleP(Y1_4, Y1_4);         // Y1_4 = 8*Y1^4
    uint64_t tempY3[4] = {Y3[0], Y3[1], Y3[2], Y3[3]};
    ModSub256(Y3, tempY3, Y1_4);     // Y3 = M*(S-X3) - 8*Y1^4

    // Z3 = 2*Y1*Z1
    ModMulP(Z3, Y1, Z1);
    ModDoubleP(Z3, Z3);
}

// ===========================================================================
// Jacobian 混合點加法 (Jacobian + Affine → Jacobian)
// (X3:Y3:Z3) = (X1:Y1:Z1) + (x2:y2:1)
// 公式 (Madd):
//   U2 = X1 + Z1^2*x2
//   S2 = Y1 + Z1^3*y2
//   H  = U2 - X1  (= Z1^2*x2 - X1 + X1 … wait, let's use standard Madd)
// Standard mixed add:
//   H  = x2*Z1^2 - X1
//   R  = y2*Z1^3 - Y1
//   X3 = R^2 - H^3 - 2*X1*H^2
//   Y3 = R*(X1*H^2 - X3) - Y1*H^3
//   Z3 = H*Z1
// ===========================================================================
__device__ void JacMixedAdd(
    uint64_t X3[4], uint64_t Y3[4], uint64_t Z3[4],
    const uint64_t X1[4], const uint64_t Y1[4], const uint64_t Z1[4],
    const uint64_t x2[4], const uint64_t y2[4])
{
    uint64_t Z1sq[4], H[4], R[4], H2[4], H3[4], tmp[4], tmp2[4];

    // Z1^2
    ModSqP(Z1sq, Z1);
    // H = x2*Z1^2 - X1
    uint64_t x2c[4]={x2[0],x2[1],x2[2],x2[3]};
    uint64_t y2c[4]={y2[0],y2[1],y2[2],y2[3]};
    uint64_t X1c[4]={X1[0],X1[1],X1[2],X1[3]};
    uint64_t Y1c[4]={Y1[0],Y1[1],Y1[2],Y1[3]};

    ModMulP(H, x2c, Z1sq);          // H = x2*Z1^2
    { uint64_t ha[4]={H[0],H[1],H[2],H[3]};
      ModSub256(H, ha, X1c); }      // H = x2*Z1^2 - X1

    // R = y2*Z1^3 - Y1
    uint64_t Z1c2[4]={Z1[0],Z1[1],Z1[2],Z1[3]};
    uint64_t Z1cu[4];
    ModMulP(Z1cu, Z1sq, Z1c2);      // Z1cu = Z1^3
    ModMulP(R, y2c, Z1cu);          // R = y2*Z1^3
    { uint64_t Ra[4]={R[0],R[1],R[2],R[3]};
      ModSub256(R, Ra, Y1c); }      // R = y2*Z1^3 - Y1

    // H^2, H^3
    ModSqP(H2, H);
    ModMulP(H3, H, H2);

    // X3 = R^2 - H^3 - 2*X1*H^2
    { uint64_t Rsq[4]; ModSqP(Rsq, R);
      { uint64_t Rsqa[4]={Rsq[0],Rsq[1],Rsq[2],Rsq[3]};
        ModSub256(tmp, Rsqa, H3); } } // tmp = R^2 - H^3
    ModMulP(tmp2, X1c, H2);         // tmp2 = X1*H^2
    ModDoubleP(tmp2, tmp2);          // tmp2 = 2*X1*H^2
    { uint64_t ta[4]={tmp[0],tmp[1],tmp[2],tmp[3]},
               tb[4]={tmp2[0],tmp2[1],tmp2[2],tmp2[3]};
      ModSub256(X3, ta, tb); }       // X3 = R^2 - H^3 - 2*X1*H^2

    // Y3 = R*(X1*H^2 - X3) - Y1*H^3
    ModMulP(tmp, X1c, H2);           // tmp = X1*H^2
    { uint64_t ta[4]={tmp[0],tmp[1],tmp[2],tmp[3]},
               X3a[4]={X3[0],X3[1],X3[2],X3[3]};
      ModSub256(tmp, ta, X3a); }     // tmp = X1*H^2 - X3
    ModMulP(Y3, R, tmp);             // Y3 = R*(X1*H^2 - X3)
    ModMulP(tmp, Y1c, H3);           // tmp = Y1*H^3
    { uint64_t Y3a[4]={Y3[0],Y3[1],Y3[2],Y3[3]},
               ta[4]={tmp[0],tmp[1],tmp[2],tmp[3]};
      ModSub256(Y3, Y3a, ta); }      // Y3 -= Y1*H^3

    // Z3 = H*Z1
    ModMulP(Z3, H, Z1c2);
}

// ===========================================================================
// Fixed-Window (w=4) 純量乘法: 計算 k*G, 結果 Jacobian 座標
// 使用預計算表 DEV_TABLE_X/Y[j] = (j+1)·G (j=0..14)
// 每次處理 k 的 4 bit 窗口：4 次 JacDouble + 1 次 JacMixedAdd
// ===========================================================================
__device__ void ScalarMultG_Jacobian(
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4],
    const uint64_t k[4])
{
    uint64_t Tx[4], Ty[4], Tz[4];
    bool started = false;

    // 從最高 4-bit 窗口 (window 63) 到最低 (window 0)
    for (int w = 63; w >= 0; w--) {
        // 倍增 4 次 (shift left by 4 bits)
        if (started) {
            for (int d = 0; d < 4; d++) {
                JacDouble(Tx, Ty, Tz, Rx, Ry, Rz);
                Rx[0]=Tx[0]; Rx[1]=Tx[1]; Rx[2]=Tx[2]; Rx[3]=Tx[3];
                Ry[0]=Ty[0]; Ry[1]=Ty[1]; Ry[2]=Ty[2]; Ry[3]=Ty[3];
                Rz[0]=Tz[0]; Rz[1]=Tz[1]; Rz[2]=Tz[2]; Rz[3]=Tz[3];
            }
        }

        // 提取 4-bit 窗口值
        int limb = w / 16;          // 哪個 uint64
        int shift = (w % 16) * 4;   // 在 uint64 中的 bit 偏移
        int wval = (int)((k[limb] >> shift) & 0xFULL);

        if (wval != 0) {
            if (!started) {
                // 第一個非零窗口：直接載入 table[wval-1]
                Rx[0]=DEV_TABLE_X[wval-1][0]; Rx[1]=DEV_TABLE_X[wval-1][1];
                Rx[2]=DEV_TABLE_X[wval-1][2]; Rx[3]=DEV_TABLE_X[wval-1][3];
                Ry[0]=DEV_TABLE_Y[wval-1][0]; Ry[1]=DEV_TABLE_Y[wval-1][1];
                Ry[2]=DEV_TABLE_Y[wval-1][2]; Ry[3]=DEV_TABLE_Y[wval-1][3];
                Rz[0]=1; Rz[1]=0; Rz[2]=0; Rz[3]=0;
                started = true;
            } else {
                // 混合加法：R += table[wval-1]
                JacMixedAdd(Tx, Ty, Tz, Rx, Ry, Rz,
                            DEV_TABLE_X[wval-1], DEV_TABLE_Y[wval-1]);
                Rx[0]=Tx[0]; Rx[1]=Tx[1]; Rx[2]=Tx[2]; Rx[3]=Tx[3];
                Ry[0]=Ty[0]; Ry[1]=Ty[1]; Ry[2]=Ty[2]; Ry[3]=Ty[3];
                Rz[0]=Tz[0]; Rz[1]=Tz[1]; Rz[2]=Tz[2]; Rz[3]=Tz[3];
            }
        }
    }
    // 若 k==0 (不應發生), R 保持未初始化 — 由呼叫端保證 k >= 1
}

// ===========================================================================
// Batch Z-inversion + 轉回仿射座標 x，再對 n 取模得 r
// 輸入：Zs[n][5] (5-limb，Z 在 Jacobian 座標)
//       Xs[n][4], Ys[n][4]
// 輸出：rs[n][4] = (X/Z^2) mod n
//
// Montgomery batch inverse 算法：
//   forward pass:  prefix[i] = Z[0]*Z[1]*...*Z[i]
//   one ModInv:    inv = prefix[n-1]^{-1}
//   backward pass: inv_Zi = inv * prefix[i-1]; inv = inv * Z[i]
// ===========================================================================
// ===========================================================================
// 並行轉換 Kernel：每個 thread 獨立執行求逆以最大化 Warp 調度效率
// ===========================================================================
__global__ void ParallelConvertToR(
    uint64_t *Xs,          // [n * 4]  Jacobian X
    uint64_t *Zs,          // [n * 5]  Jacobian Z (5 limbs)
    uint64_t *rs,          // [n * 4]  output r values
    int       n)
{
    __shared__ uint64_t sh_prefixes[256 * 4]; 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ltid = threadIdx.x;

    uint64_t z[4] = {1, 0, 0, 0};
    bool valid = tid < n;

    if (valid) {
        z[0] = Zs[tid * 5 + 0];
        z[1] = Zs[tid * 5 + 1];
        z[2] = Zs[tid * 5 + 2];
        z[3] = Zs[tid * 5 + 3];
        // 處理 Z=0 的情況 (無窮遠點)
        if (z[0] == 0 && z[1] == 0 && z[2] == 0 && z[3] == 0) {
            z[0] = 1; // 暫時設為 1 以免 Batch Inverse 出錯，最後再改回 r=0
        }
    }

    // Step 1: 前綴乘積
    sh_prefixes[ltid * 4 + 0] = z[0];
    sh_prefixes[ltid * 4 + 1] = z[1];
    sh_prefixes[ltid * 4 + 2] = z[2];
    sh_prefixes[ltid * 4 + 3] = z[3];

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        uint64_t tmp[4];
        if (ltid >= stride) {
            uint64_t a[4], b[4];
            a[0] = sh_prefixes[ltid * 4 + 0]; a[1] = sh_prefixes[ltid * 4 + 1];
            a[2] = sh_prefixes[ltid * 4 + 2]; a[3] = sh_prefixes[ltid * 4 + 3];
            b[0] = sh_prefixes[(ltid - stride) * 4 + 0]; b[1] = sh_prefixes[(ltid - stride) * 4 + 1];
            b[2] = sh_prefixes[(ltid - stride) * 4 + 2]; b[3] = sh_prefixes[(ltid - stride) * 4 + 3];
            ModMulP(tmp, a, b);
        }
        __syncthreads();
        if (ltid >= stride) {
            sh_prefixes[ltid * 4 + 0] = tmp[0]; sh_prefixes[ltid * 4 + 1] = tmp[1];
            sh_prefixes[ltid * 4 + 2] = tmp[2]; sh_prefixes[ltid * 4 + 3] = tmp[3];
        }
    }

    // Step 2: 單個 ModInv
    __shared__ uint64_t sh_inv_all[5];
    if (ltid == blockDim.x - 1) {
        sh_inv_all[0] = sh_prefixes[ltid * 4 + 0];
        sh_inv_all[1] = sh_prefixes[ltid * 4 + 1];
        sh_inv_all[2] = sh_prefixes[ltid * 4 + 2];
        sh_inv_all[3] = sh_prefixes[ltid * 4 + 3];
        sh_inv_all[4] = 0;
        _ModInv(sh_inv_all);
    }
    __syncthreads();

    if (!valid) return;

    // Step 3: 計算個別 invZ
    if (ltid == 0) {
        // 第一個線程的 invZ 就已經是 inv_Pn * S_1 (因為 P_0 = z_0) ? No.
    } 
    
    // ... (rest of the logic remains same, just removing the unused block above)
    
    // 考慮到 BlockDim 只有 256，最有效率的是「後向前綴乘積」
    __shared__ uint64_t sh_suffixes[256 * 4];
    sh_suffixes[ltid * 4 + 0] = z[0];
    sh_suffixes[ltid * 4 + 1] = z[1];
    sh_suffixes[ltid * 4 + 2] = z[2];
    sh_suffixes[ltid * 4 + 3] = z[3];

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        uint64_t tmp[4];
        if (ltid + stride < blockDim.x) {
            uint64_t a[4], b[4];
            a[0] = sh_suffixes[ltid * 4 + 0]; a[1] = sh_suffixes[ltid * 4 + 1];
            a[2] = sh_suffixes[ltid * 4 + 2]; a[3] = sh_suffixes[ltid * 4 + 3];
            b[0] = sh_suffixes[(ltid + stride) * 4 + 0]; b[1] = sh_suffixes[(ltid + stride) * 4 + 1];
            b[2] = sh_suffixes[(ltid + stride) * 4 + 2]; b[3] = sh_suffixes[(ltid + stride) * 4 + 3];
            ModMulP(tmp, a, b);
        }
        __syncthreads();
        if (ltid + stride < blockDim.x) {
            sh_suffixes[ltid * 4 + 0] = tmp[0]; sh_suffixes[ltid * 4 + 1] = tmp[1];
            sh_suffixes[ltid * 4 + 2] = tmp[2]; sh_suffixes[ltid * 4 + 3] = tmp[3];
        }
    }
    __syncthreads();

    // 現在：
    // sh_prefixes[i] = z_0 * ... * z_i
    // sh_suffixes[i] = z_i * ... * z_{n-1}
    // inv_Zi = inv_Pn * (z_0 * ... * z_{i-1}) * (z_{i+1} * ... * z_{n-1})
    
    uint64_t final_inv[4];
    uint64_t base_inv[4] = {sh_inv_all[0], sh_inv_all[1], sh_inv_all[2], sh_inv_all[3]};
    
    uint64_t part[4] = {1, 0, 0, 0};
    if (ltid > 0) {
        part[0] = sh_prefixes[(ltid - 1) * 4 + 0];
        part[1] = sh_prefixes[(ltid - 1) * 4 + 1];
        part[2] = sh_prefixes[(ltid - 1) * 4 + 2];
        part[3] = sh_prefixes[(ltid - 1) * 4 + 3];
    }
    ModMulP(final_inv, base_inv, part);
    
    if (ltid < blockDim.x - 1) {
        uint64_t suff[4];
        suff[0] = sh_suffixes[(ltid + 1) * 4 + 0];
        suff[1] = sh_suffixes[(ltid + 1) * 4 + 1];
        suff[2] = sh_suffixes[(ltid + 1) * 4 + 2];
        suff[3] = sh_suffixes[(ltid + 1) * 4 + 3];
        ModMulP(final_inv, final_inv, suff);
    }
    
    // final_inv 現在就是 z_i^{-1} mod P
    
    // 檢查原本是否為無窮遠點
    if (Zs[tid*5+0]==0 && Zs[tid*5+1]==0 && Zs[tid*5+2]==0 && Zs[tid*5+3]==0) {
        rs[tid*4+0] = rs[tid*4+1] = rs[tid*4+2] = rs[tid*4+3] = 0;
        return;
    }

    uint64_t invZ2[5];
    uint64_t invZ5[5] = {final_inv[0], final_inv[1], final_inv[2], final_inv[3], 0};
    _ModMult(invZ2, invZ5, invZ5);
    invZ2[4] = 0;

    uint64_t Xi[4] = {Xs[tid*4+0], Xs[tid*4+1], Xs[tid*4+2], Xs[tid*4+3]};
    uint64_t X5[5] = {Xi[0], Xi[1], Xi[2], Xi[3], 0};
    uint64_t Xa5[5];
    _ModMult(Xa5, X5, invZ2);

    uint64_t xa[4] = {Xa5[0], Xa5[1], Xa5[2], Xa5[3]};
    uint64_t local_N[4] = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
    
    int ge = (xa[3]>local_N[3]) ||
             (xa[3]==local_N[3] && xa[2]>local_N[2]) ||
             (xa[3]==local_N[3] && xa[2]==local_N[2] && xa[1]>local_N[1]) ||
             (xa[3]==local_N[3] && xa[2]==local_N[2] && xa[1]==local_N[1] && xa[0]>=local_N[0]);
    
    uint64_t *ri = rs + tid * 4;
    if (ge) {
        ModSub256(ri, xa, local_N);
    } else {
        ri[0]=xa[0]; ri[1]=xa[1]; ri[2]=xa[2]; ri[3]=xa[3];
    }
}

// ===========================================================================
// ===========================================================================
// GPU Bloom Filter Kernels
// ===========================================================================

// 用於 GPU Bloom Filter 的 Hash 函數 (針對 11-char hex 字符串)
__device__ __forceinline__ uint64_t gpu_hash_hex11(const char* hex_str, uint32_t seed) {
    uint64_t h = seed;
    for (int i = 0; i < 11; i++) {
        char c = hex_str[i];
        uint8_t v = (c >= '0' && c <= '9') ? (c - '0') : (c - 'a' + 10);
        h = h * 31 + v;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
    }
    return h;
}

// GPU Kernel: 將 hex 字符串批量添加到 Bloom Filter
// 每個 thread 負責一個 key
__global__ void gpu_bloom_add_batch(uint64_t *bloom_bits, uint64_t num_bits, int num_hashes, 
                                   const char* hex_keys, int num_keys, int str_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    const char* key = hex_keys + tid * str_len;
    
    for (int i = 0; i < num_hashes; i++) {
        uint64_t h = gpu_hash_hex11(key, i) % num_bits;
        size_t word_idx = h / 64;
        size_t bit_idx = h % 64;
        atomicOr((unsigned long long*)&bloom_bits[word_idx], 1ULL << bit_idx);
    }
}

// GPU Kernel: 查詢 Bloom Filter（批量）
// 每個 thread 負責一個 key，寫結果到 results 陣列
__global__ void gpu_bloom_query_batch(uint64_t *bloom_bits, uint64_t num_bits, int num_hashes,
                                      const char* hex_keys, int num_keys, int str_len, uint8_t* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    const char* key = hex_keys + tid * str_len;
    
    bool possibly_contains = true;
    for (int i = 0; i < num_hashes; i++) {
        uint64_t h = gpu_hash_hex11(key, i) % num_bits;
        size_t word_idx = h / 64;
        size_t bit_idx = h % 64;
        if (!(bloom_bits[word_idx] & (1ULL << bit_idx))) {
            possibly_contains = false;
            break;
        }
    }
    
    results[tid] = possibly_contains ? 1 : 0;  // 1: 可能存在, 0: 確定不存在
}

// 主 GPU Kernel：每個 thread 計算一個 k 的 k*G (Jacobian 座標)
// k_in:  [n * 4]  輸入 k 值 (4 × uint64, 小端)
// Xs:    [n * 4]  輸出 Jacobian X
// Zs:    [n * 5]  輸出 Jacobian Z
// ===========================================================================
// ===========================================================================
// 主 GPU Kernel：每個 thread 計算一個 k 的 k*G (Jacobian 座標)
// k_start_limbs: 當前 batch 的起始 k 值
// ===========================================================================
__global__ void ComputeKG_Jacobian(
    uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3,
    uint64_t *Xs,
    uint64_t *Ys,
    uint64_t *Zs,
    int      n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 計算當前 thread 的 k = k_start + tid
    uint64_t k[4];
    uint64_t carry = (uint64_t)tid;
    
    // Limb 0
    k[0] = k0 + carry;
    carry = (k[0] < k0) ? 1 : 0;
    // Limb 1
    k[1] = k1 + carry;
    carry = (k[1] < k1) ? 1 : 0;
    // Limb 2
    k[2] = k2 + carry;
    carry = (k[2] < k2) ? 1 : 0;
    // Limb 3
    k[3] = k3 + carry;

    uint64_t Rx[4], Ry[4], Rz[4];
    ScalarMultG_Jacobian(Rx, Ry, Rz, k);

    uint64_t *Xo = Xs + tid * 4;
    uint64_t *Yo = Ys + tid * 4;
    uint64_t *Zo = Zs + tid * 5;

    Xo[0]=Rx[0]; Xo[1]=Rx[1]; Xo[2]=Rx[2]; Xo[3]=Rx[3];
    Yo[0]=Ry[0]; Yo[1]=Ry[1]; Yo[2]=Ry[2]; Yo[3]=Ry[3];
    Zo[0]=Rz[0]; Zo[1]=Rz[1]; Zo[2]=Rz[2]; Zo[3]=Rz[3]; Zo[4]=0;
}

// ===========================================================================
// Host 端工具函式
// ===========================================================================
static void printHex256(const char *label, const uint64_t v[4]) {
    printf("%s = %016llx%016llx%016llx%016llx\n", label,
           (unsigned long long)v[3], (unsigned long long)v[2],
           (unsigned long long)v[1], (unsigned long long)v[0]);
}

// ===========================================================================
// Fast Hex Conversion
// ===========================================================================
static const char HEX_CHARS[] = "0123456789abcdef";

static void to_hex256(char *dest, const uint64_t v[4]) {
    for (int i = 0; i < 4; i++) {
        uint64_t val = v[3 - i];
        for (int j = 15; j >= 0; j--) {
            dest[i * 16 + j] = HEX_CHARS[val & 0xf];
            val >>= 4;
        }
    }
    dest[64] = '\0';
}

static void to_hex44(char *dest, uint64_t v0) {
    uint64_t val = v0 & 0xFFFFFFFFFFFULL;
    for (int j = 10; j >= 0; j--) {
        dest[j] = HEX_CHARS[val & 0xf];
        val >>= 4;
    }
    dest[11] = '\0';
}

static std::string hex256(const uint64_t v[4]) {
    char buf[65];
    to_hex256(buf, v);
    return std::string(buf);
}

// ===========================================================================
// Thread-Safe Task Queue for Pipelining
// ===========================================================================
struct BatchTask {
    int stream_idx;
    uint64_t k_start[4];
    uint64_t* r_ptr;
    int n;
};

template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv;
public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(m);
        q.push(val);
        cv.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this]{ return !q.empty(); });
        T val = q.front();
        q.pop();
        return val;
    }
    size_t size() {
        std::lock_guard<std::mutex> lock(m);
        return q.size();
    }
};

struct MatchData {
    std::string k;
    std::string r;
    bool exit_signal = false;
};

SafeQueue<MatchData> g_match_queue;
std::atomic<unsigned long long> g_stat_total_scanned(0);
std::atomic<unsigned long long> g_stat_bloom_filtered(0);
std::atomic<unsigned long long> g_stat_redis_queried(0);

void MongoMatchLogger(const char* mongo_uri, const char* redis_ip, int redis_port) {
    mongoc_client_t *client = mongoc_client_new(mongo_uri);
    if (!client) {
        fprintf(stderr, "MatchLogger: Failed to parse MongoDB URI\n");
        return;
    }
    mongoc_collection_t *match_col = mongoc_client_get_collection(client, "ecdsa", "match");
    mongoc_collection_t *btc_col = mongoc_client_get_collection(client, "ecdsa", "btc");

    // Connect to Redis for managing the matched keys set
    redisContext *redis_ctx = redisConnect(redis_ip, redis_port);
    if (redis_ctx == nullptr || redis_ctx->err) {
        fprintf(stderr, "MatchLogger: Redis Connection Error\n");
        if (redis_ctx) redisFree(redis_ctx);
        // Continue without Redis if it fails, or you could exit. We will continue and just skip Redis ops.
    } else {
        redisReply *reply = (redisReply *)redisCommand(redis_ctx, "SELECT 0");
        if (reply) freeReplyObject(reply);
    }

    while (true) {
        MatchData data = g_match_queue.pop();
        if (data.exit_signal) break;

        std::string ks = data.k;
        std::string rs = data.r;
        std::string redis_val = rs + ":" + ks;

        // 1. Write the found match (r:k) into Redis Set
        if (redis_ctx && !redis_ctx->err) {
            redisReply *reply = (redisReply *)redisCommand(redis_ctx, "SADD matched_candidates %s", redis_val.c_str());
            if (reply) freeReplyObject(reply);
        }

        // 2. Query MongoDB btc collection
        std::string r0 = "0" + rs;
        std::string r00 = "00" + rs;

        // Query using $in: { "_id": { "$in": [rs, r0, r00] } }
        bson_t *query = bson_new();
        bson_t in_child;
        BSON_APPEND_DOCUMENT_BEGIN(query, "_id", &in_child);
        bson_t in_array;
        BSON_APPEND_ARRAY_BEGIN(&in_child, "$in", &in_array);
        BSON_APPEND_UTF8(&in_array, "0", rs.c_str());
        BSON_APPEND_UTF8(&in_array, "1", r0.c_str());
        BSON_APPEND_UTF8(&in_array, "2", r00.c_str());
        bson_append_array_end(&in_child, &in_array);
        bson_append_document_end(query, &in_child);

        mongoc_cursor_t *cursor = mongoc_collection_find_with_opts(btc_col, query, NULL, NULL);
        const bson_t *found_btc_doc;
        bool is_precise_match = mongoc_cursor_next(cursor, &found_btc_doc);
        
        if (is_precise_match) {
            // Precise match found! Write to MongoDB match collection.
            bson_t *doc = bson_new();
            BSON_APPEND_UTF8(doc, "r", rs.c_str());
            BSON_APPEND_UTF8(doc, "k", ks.c_str());
            BSON_APPEND_UTF8(doc, "found", "y");
            BSON_APPEND_TIME_T(doc, "time", time(NULL));
            
            bson_error_t error;
            if (!mongoc_collection_insert_one(match_col, doc, NULL, NULL, &error)) {
                fprintf(stderr, "MongoDB Insert Error: %s\n", error.message);
            }
            bson_destroy(doc);
            
            printf("               [PRECISE MATCH FOUND IN BG!] Inserted into match col for r=%s\n", rs.c_str());
        } else {
            // Not a precise match. Remove from Redis Set.
            if (redis_ctx && !redis_ctx->err) {
                redisReply *reply = (redisReply *)redisCommand(redis_ctx, "SREM matched_candidates %s", redis_val.c_str());
                if (reply) freeReplyObject(reply);
            }
        }
        
        mongoc_cursor_destroy(cursor);
        bson_destroy(query);
    }

    if (redis_ctx) redisFree(redis_ctx);
    mongoc_collection_destroy(match_col);
    mongoc_collection_destroy(btc_col);
    mongoc_client_destroy(client);
}

// ===========================================================================
// Hybrid Scanner Logic (Redis + MongoDB)
// ===========================================================================
class HybridScanner {
public:
    redisContext *redis_ctx = nullptr;
    char* tag_buffer = nullptr;
    uint8_t* bloom_host_results = nullptr;
    int current_n = 0;

        HybridScanner(const char *redis_ip, int redis_port,
                                    uint64_t *dev_bloom, uint64_t bloom_bits_count, int bloom_hash_count,
                                    std::atomic<bool> *bloom_ready)
                : dev_bloom_bits(dev_bloom),
                    bloom_num_bits(bloom_bits_count),
                    bloom_num_hashes(bloom_hash_count),
                    bloom_ready_flag(bloom_ready) {
        // Connect to Redis
        redis_ctx = redisConnect(redis_ip, redis_port);
        if (redis_ctx == nullptr || redis_ctx->err) {
            if (redis_ctx) printf("Redis Connection Error: %s\n", redis_ctx->errstr);
            else printf("Redis Connection Error: Can't allocate redis context\n");
            exit(1);
        }
        redisReply *reply = (redisReply *)redisCommand(redis_ctx, "SELECT 0");
        freeReplyObject(reply);

        // Per-worker GPU buffers for Bloom prefilter.
        const int sub_batch_size = 50000;
        checkCuda(cudaMalloc(&dev_bloom_results, sub_batch_size * sizeof(uint8_t)), "scanner bloom results malloc");
        checkCuda(cudaMalloc(&dev_hex_buffer, sub_batch_size * 12), "scanner hex buffer malloc");
    }

    ~HybridScanner() {
        if (redis_ctx) redisFree(redis_ctx);
        if (tag_buffer) delete[] tag_buffer;
        if (bloom_host_results) delete[] bloom_host_results;
        if (dev_bloom_results) cudaFree(dev_bloom_results);
        if (dev_hex_buffer) cudaFree(dev_hex_buffer);
    }

    // GPU Bloom Filter 成員
    uint64_t *dev_bloom_bits = nullptr;
    uint64_t bloom_num_bits = 0;
    int bloom_num_hashes = 0;
    std::atomic<bool> *bloom_ready_flag = nullptr;
    
    uint8_t *dev_bloom_results = nullptr;  // GPU結果緩衝區
    char *dev_hex_buffer = nullptr;         // GPU Hex字符串緩衝區

    void checkMatchBatch(const uint64_t k_start[4], const uint64_t *rs_ptr, int n) {
        if (n <= 0) return;

        const int sub_batch_size = 50000;
        
        if (sub_batch_size > current_n) {
            if (tag_buffer) delete[] tag_buffer;
            if (bloom_host_results) delete[] bloom_host_results;
            tag_buffer = new char[sub_batch_size * 12];
            bloom_host_results = new uint8_t[sub_batch_size];
            current_n = sub_batch_size;
        }

        for (int start = 0; start < n; start += sub_batch_size) {
            int current_batch = (n - start < sub_batch_size) ? (n - start) : sub_batch_size;
            g_stat_total_scanned.fetch_add((unsigned long long)current_batch, std::memory_order_relaxed);

            for (int i = 0; i < current_batch; i++) {
                to_hex44(tag_buffer + i * 12, rs_ptr[(start + i) * 4]);
            }

            std::vector<int> candidate_indices;
            bool bloom_ready = (bloom_ready_flag != nullptr) && bloom_ready_flag->load();
            if (bloom_ready) {
                // GPU Bloom prefilter: only probable hits go to Redis.
                int bloom_threads = 256;
                int bloom_blocks = (current_batch + bloom_threads - 1) / bloom_threads;
                checkCuda(cudaMemcpy(dev_hex_buffer, tag_buffer, current_batch * 12, cudaMemcpyHostToDevice),
                          "scanner bloom hex memcpy");
                gpu_bloom_query_batch<<<bloom_blocks, bloom_threads>>>(
                    dev_bloom_bits,
                    bloom_num_bits,
                    bloom_num_hashes,
                    dev_hex_buffer,
                    current_batch,
                    12,
                    dev_bloom_results);
                checkCuda(cudaGetLastError(), "scanner bloom query launch");
                checkCuda(cudaMemcpy(bloom_host_results, dev_bloom_results, current_batch * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                          "scanner bloom results memcpy");

                candidate_indices.reserve(current_batch / 16 + 1);
                for (int i = 0; i < current_batch; i++) {
                    if (bloom_host_results[i] == 1) candidate_indices.push_back(i);
                }
            } else {
                // Bloom not ready yet: fallback to Redis-only path to avoid false negatives.
                candidate_indices.reserve(current_batch);
                for (int i = 0; i < current_batch; i++) candidate_indices.push_back(i);
            }

            if (candidate_indices.empty()) {
                g_stat_bloom_filtered.fetch_add((unsigned long long)current_batch, std::memory_order_relaxed);
                continue;
            }

            if (candidate_indices.size() < (size_t)current_batch) {
                g_stat_bloom_filtered.fetch_add((unsigned long long)(current_batch - (int)candidate_indices.size()), std::memory_order_relaxed);
            }
            g_stat_redis_queried.fetch_add((unsigned long long)candidate_indices.size(), std::memory_order_relaxed);

            std::vector<const char*> argv;
            std::vector<size_t> argvlen;
            argv.reserve(candidate_indices.size() + 2);
            argvlen.reserve(candidate_indices.size() + 2);

            argv.push_back("SMISMEMBER");
            argvlen.push_back(10);
            argv.push_back("mongo_keys");
            argvlen.push_back(10);
            for (size_t i = 0; i < candidate_indices.size(); i++) {
                int idx = candidate_indices[i];
                argv.push_back(tag_buffer + idx * 12);
                argvlen.push_back(11);
            }

            redisReply *reply = (redisReply *)redisCommandArgv(redis_ctx, argv.size(), argv.data(), argvlen.data());
            
            if (reply == nullptr) {
                fprintf(stderr, "Redis error: %s\n", redis_ctx->errstr);
                continue;
            }

            if (reply->type == REDIS_REPLY_ARRAY) {
                for (size_t j = 0; j < reply->elements; j++) {
                    if (reply->element[j]->integer == 1) {
                        int original_idx = candidate_indices[j];
                        // Redis 命中！還原 k 值
                        uint64_t k_match[4];
                        uint64_t carry = (uint64_t)(start + original_idx);
                        for (int limb = 0; limb < 4; limb++) {
                            k_match[limb] = k_start[limb] + carry;
                            carry = (k_match[limb] < k_start[limb]) ? 1 : 0;
                        }

                        std::string ks = hex256(k_match);
                        std::string rs = hex256(rs_ptr + (start + original_idx) * 4);

                        // 丟入背景佇列處理 MongoDB
                        MatchData mdata;
                        mdata.k = ks;
                        mdata.r = rs;
                        g_match_queue.push(mdata);
                    }
                }
            }
            freeReplyObject(reply);
        }
    }
};

static void parse256(uint64_t r[4], const char *s) {
    memset(r, 0, 32);
    if (!s) return;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
        int len = strlen(s);
        for (int i = 0; i < len; i++) {
            int val = 0;
            if (s[i] >= '0' && s[i] <= '9') val = s[i] - '0';
            else if (s[i] >= 'a' && s[i] <= 'f') val = s[i] - 'a' + 10;
            else if (s[i] >= 'A' && s[i] <= 'F') val = s[i] - 'A' + 10;
            else break;
            uint64_t carry = val;
            for (int j = 0; j < 4; j++) {
                uint64_t next_carry = r[j] >> 60;
                r[j] = (r[j] << 4) + carry;
                carry = next_carry;
            }
        }
    } else {
        for (int i = 0; s[i] != '\0'; i++) {
            if (s[i] < '0' || s[i] > '9') break;
            int digit = s[i] - '0';
            uint64_t carry = digit;
            for (int j = 0; j < 4; j++) {
                __uint128_t prod = (__uint128_t)r[j] * 10 + carry;
                r[j] = (uint64_t)prod;
                carry = (uint64_t)(prod >> 64);
            }
        }
    }
}

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ===========================================================================
// CPU 端產生密碼學安全的隨機 256-bit k 值 (使用 /dev/urandom)
// 保證 k ∈ [1, n-1]（n = secp256k1 群階數）
// ===========================================================================
static void generate_random_k(uint64_t k[4]) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f) { perror("fopen /dev/urandom"); exit(1); }
    while (true) {
        if (fread(k, 32, 1, f) != 1) {
            perror("fread /dev/urandom");
            fclose(f);
            exit(1);
        }
        // 檢查 k == 0
        if (k[0] == 0 && k[1] == 0 && k[2] == 0 && k[3] == 0)
            continue;
        // 檢查 k >= n (HOST_N)
        bool ge_n = (k[3] > HOST_N[3]) ||
                    (k[3] == HOST_N[3] && k[2] > HOST_N[2]) ||
                    (k[3] == HOST_N[3] && k[2] == HOST_N[2] && k[1] > HOST_N[1]) ||
                    (k[3] == HOST_N[3] && k[2] == HOST_N[2] && k[1] == HOST_N[1] && k[0] >= HOST_N[0]);
        if (ge_n)
            continue;
        break;  // k ∈ [1, n-1]，有效
    }
    fclose(f);
}

// ===========================================================================
// CPU-side 256-bit 模算術 (僅用於啟動時預計算 j·G 表)
// ===========================================================================
static int host_cmp256_ge(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;
}

static uint64_t host_add256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    __uint128_t c = 0;
    for (int i = 0; i < 4; i++) { c += (__uint128_t)a[i] + b[i]; r[i] = (uint64_t)c; c >>= 64; }
    return (uint64_t)c;
}

static uint64_t host_sub256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    int64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        __int128_t d = (__int128_t)a[i] - b[i] - borrow;
        r[i] = (uint64_t)d; borrow = (d < 0) ? 1 : 0;
    }
    return (uint64_t)borrow;
}

static void h_mod_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t c = host_add256(r, a, b);
    if (c || host_cmp256_ge(r, HOST_P)) host_sub256(r, r, HOST_P);
}

static void h_mod_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t b_ = host_sub256(r, a, b);
    if (b_) host_add256(r, r, HOST_P);
}

static void h_mod_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t t[8] = {0};
    for (int i = 0; i < 4; i++) {
        __uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            carry += (__uint128_t)a[i] * b[j] + t[i + j];
            t[i + j] = (uint64_t)carry; carry >>= 64;
        }
        t[i + 4] = (uint64_t)carry;
    }
    // P = 2^256 - C, C = 0x1000003D1
    const uint64_t C = 0x1000003D1ULL;
    // Round 1: tmp = t_low + t_high * C
    __uint128_t carry = 0;
    uint64_t tmp[5];
    for (int i = 0; i < 4; i++) {
        carry += (__uint128_t)t[i + 4] * C + t[i];
        tmp[i] = (uint64_t)carry; carry >>= 64;
    }
    tmp[4] = (uint64_t)carry;
    // Round 2: r = tmp[0..3] + tmp[4] * C
    carry = (__uint128_t)tmp[4] * C;
    for (int i = 0; i < 4; i++) {
        carry += tmp[i]; r[i] = (uint64_t)carry; carry >>= 64;
    }
    if (carry || host_cmp256_ge(r, HOST_P)) host_sub256(r, r, HOST_P);
}

static void h_mod_sqr(uint64_t r[4], const uint64_t a[4]) { h_mod_mul(r, a, a); }

// a^{-1} mod P via Fermat: a^{P-2} mod P (square-and-multiply)
static void h_mod_inv(uint64_t r[4], const uint64_t a[4]) {
    // P-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    uint64_t pm2[4] = { HOST_P[0] - 2, HOST_P[1], HOST_P[2], HOST_P[3] };
    uint64_t base[4] = { a[0], a[1], a[2], a[3] };
    r[0] = 1; r[1] = 0; r[2] = 0; r[3] = 0;
    for (int w = 0; w < 4; w++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((pm2[w] >> bit) & 1) h_mod_mul(r, r, base);
            h_mod_sqr(base, base);
        }
    }
}

// 仿射座標點倍增: (x3,y3) = 2*(px,py)
static void h_point_double(uint64_t x3[4], uint64_t y3[4],
                           const uint64_t px[4], const uint64_t py[4]) {
    // lambda = 3*x^2 / (2*y)  (a=0 for secp256k1)
    uint64_t x2[4], num[4], den[4], inv_den[4], lam[4], lam2[4], tmp[4];
    h_mod_sqr(x2, px);           // x^2
    h_mod_add(num, x2, x2);     // 2*x^2
    h_mod_add(num, num, x2);    // 3*x^2
    h_mod_add(den, py, py);     // 2*y
    h_mod_inv(inv_den, den);
    h_mod_mul(lam, num, inv_den);
    h_mod_sqr(lam2, lam);       // lambda^2
    h_mod_add(tmp, px, px);     // 2*x
    h_mod_sub(x3, lam2, tmp);   // x3 = lambda^2 - 2*x
    h_mod_sub(tmp, px, x3);     // x - x3
    h_mod_mul(y3, lam, tmp);    // lambda*(x - x3)
    h_mod_sub(y3, y3, py);      // y3 = lambda*(x - x3) - y
}

// 仿射座標點加法: (x3,y3) = (p1x,p1y) + (p2x,p2y)
static void h_point_add(uint64_t x3[4], uint64_t y3[4],
                        const uint64_t p1x[4], const uint64_t p1y[4],
                        const uint64_t p2x[4], const uint64_t p2y[4]) {
    uint64_t dy[4], dx[4], inv_dx[4], lam[4], lam2[4], tmp[4];
    h_mod_sub(dy, p2y, p1y);
    h_mod_sub(dx, p2x, p1x);
    h_mod_inv(inv_dx, dx);
    h_mod_mul(lam, dy, inv_dx);
    h_mod_sqr(lam2, lam);
    h_mod_sub(x3, lam2, p1x);
    h_mod_sub(x3, x3, p2x);
    h_mod_sub(tmp, p1x, x3);
    h_mod_mul(y3, lam, tmp);
    h_mod_sub(y3, y3, p1y);
}

// 預計算 table[j] = (j+1)·G (j=0..14) 的仿射座標
static void precompute_G_table(uint64_t tx[15][4], uint64_t ty[15][4]) {
    // table[0] = 1·G
    memcpy(tx[0], HOST_GX, 32);
    memcpy(ty[0], HOST_GY, 32);
    // table[1] = 2·G
    h_point_double(tx[1], ty[1], tx[0], ty[0]);
    // table[j] = table[j-1] + G  for j >= 2
    for (int j = 2; j < 15; j++) {
        h_point_add(tx[j], ty[j], tx[j - 1], ty[j - 1], tx[0], ty[0]);
    }
    printf("[Precompute] 已計算 15 個 j·G 仿射座標點\n");
}


int main(int argc, char **argv) {
    int n = 1048576; 
    bool infinite = false;

    if (argc > 1) {
        n = atoi(argv[1]);
        if (n == 0) {
            n = 1048576;
            infinite = true;
        }
    }
    if (n <= 0) n = 1048576;

    printf("=== CalcR_GPU: secp256k1 r = x(k*G) mod n (Random K Mode) ===\n");
    printf("批次大小: %d\n", n);
    printf("k 值由 CPU /dev/urandom 隨機產生\n");
    
    // Initialize MongoDB Driver (Global)
    mongoc_init();

    checkCuda(cudaMemcpyToSymbol(DEV_P,  HOST_P,  32), "cpyP");
    checkCuda(cudaMemcpyToSymbol(DEV_N,  HOST_N,  32), "cpyN");
    checkCuda(cudaMemcpyToSymbol(DEV_GX, HOST_GX, 32), "cpyGx");
    checkCuda(cudaMemcpyToSymbol(DEV_GY, HOST_GY, 32), "cpyGy");

    // 預計算 j·G (j=1..15) 並上傳到 GPU constant memory
    uint64_t table_x[15][4], table_y[15][4];
    precompute_G_table(table_x, table_y);
    checkCuda(cudaMemcpyToSymbol(DEV_TABLE_X, table_x, sizeof(table_x)), "cpyTableX");
    checkCuda(cudaMemcpyToSymbol(DEV_TABLE_Y, table_y, sizeof(table_y)), "cpyTableY");

    // -----------------------------------------------------------------------
    // 分配 Ring Buffer 與 CUDA Streams (增加到 32 個以對抗 Redis 延遲)
    // -----------------------------------------------------------------------
    const int num_streams = 32;
    cudaStream_t streams[num_streams];
    uint64_t *d_X[num_streams], *d_Y[num_streams], *d_Z[num_streams], *d_r[num_streams];
    uint64_t *h_r[num_streams], *h_k[num_streams];

    size_t X_bytes = (size_t)n * 4 * sizeof(uint64_t);
    size_t Z_bytes = (size_t)n * 5 * sizeof(uint64_t);

    for (int i = 0; i < num_streams; i++) {
        checkCuda(cudaStreamCreate(&streams[i]), "stream create");
        checkCuda(cudaMalloc(&d_X[i], X_bytes), "malloc X");
        checkCuda(cudaMalloc(&d_Y[i], X_bytes), "malloc Y");
        checkCuda(cudaMalloc(&d_Z[i], Z_bytes), "malloc Z");
        checkCuda(cudaMalloc(&d_r[i], X_bytes), "malloc r");
        checkCuda(cudaHostAlloc(&h_r[i], X_bytes, cudaHostAllocPortable), "hostAlloc r");
        checkCuda(cudaHostAlloc(&h_k[i], X_bytes, cudaHostAllocPortable), "hostAlloc k");
    }

    // -----------------------------------------------------------------------
    // 多執行緒 Redis Worker 池 (增加到 16 個 Worker)
    // -----------------------------------------------------------------------
    const int num_workers = 16;

    // -----------------------------------------------------------------------
    // GPU Bloom Filter 初始化 (1.5B keys @ 0.05% FPR)
    // -----------------------------------------------------------------------
    uint64_t expected_mongo_keys = 1500000000ULL;   // 1.5 billion keys
    double fpr = 0.0005;                            // 0.05% false positive rate
    const size_t bloom_budget_mb = 8000;            // VRAM budget for Bloom bits
    
    // 計算最優 bit 數：m = -n*ln(p) / (ln(2)^2)
    uint64_t ideal_bits = (uint64_t)(-(double)expected_mongo_keys * log(fpr) / (log(2) * log(2)));
    uint64_t budget_bits = (uint64_t)bloom_budget_mb * 1024ULL * 1024ULL * 8ULL;
    uint64_t bloom_num_bits = (ideal_bits > budget_bits) ? budget_bits : ideal_bits;
    
    // 計算最優 hash 函數數：k = (m/n) * ln(2)
    int bloom_num_hashes = (int)llround((double)bloom_num_bits / expected_mongo_keys * log(2));
    if (bloom_num_hashes < 1) bloom_num_hashes = 1;
    if (bloom_num_hashes > 20) bloom_num_hashes = 20;
    
    printf("[Bloom Filter] 初始化: %llu bits (%.2f MB), %d hash functions, FPR=%.4f%%\n",
        (unsigned long long)bloom_num_bits,
        (double)bloom_num_bits / 8.0 / 1024.0 / 1024.0,
        bloom_num_hashes, fpr * 100.0);
    if (ideal_bits > budget_bits) {
        printf("[Bloom Filter] 已套用 %zu MB 顯存上限（理想需求 %.2f MB）\n",
            bloom_budget_mb,
            (double)ideal_bits / 8.0 / 1024.0 / 1024.0);
    }
    
    // 分配 GPU 記憶體用於 Bloom Filter
    size_t bloom_array_size = (bloom_num_bits + 63) / 64;
    uint64_t *dev_bloom_bits = nullptr;
    checkCuda(cudaMalloc(&dev_bloom_bits, bloom_array_size * sizeof(uint64_t)), "Bloom malloc");
    checkCuda(cudaMemset(dev_bloom_bits, 0, bloom_array_size * sizeof(uint64_t)), "Bloom memset");
    
    printf("[Bloom Filter] 共分配 %.2f MB 的 GPU 記憶體\n",
           (double)bloom_array_size * sizeof(uint64_t) / 1024.0 / 1024.0);
    
    // -----------------------------------------------------------------------
    // Bloom Filter 快取檔案: 優先從檔案載入，否則從 Redis 建立後存檔
    // 檔案格式: [header 32 bytes] + [bloom bit array]
    // header: magic(8) + version(4) + num_bits(8) + num_hashes(4) + key_count(8)
    // -----------------------------------------------------------------------
    const char *bloom_cache_path = "bloom_cache.bin";
    std::atomic<bool> bloom_loader_done(false);
    std::atomic<bool> bloom_filter_ready(false);
    std::atomic<uint64_t> bloom_loaded_count(0);
    
    // 嘗試從快取檔案載入
    bool loaded_from_cache = false;
    {
        FILE *cf = fopen(bloom_cache_path, "rb");
        if (cf) {
            // 讀取 header
            uint64_t magic = 0;
            uint32_t version = 0;
            uint64_t file_num_bits = 0;
            uint32_t file_num_hashes = 0;
            uint64_t file_key_count = 0;
            
            bool header_ok = (fread(&magic, 8, 1, cf) == 1) &&
                             (fread(&version, 4, 1, cf) == 1) &&
                             (fread(&file_num_bits, 8, 1, cf) == 1) &&
                             (fread(&file_num_hashes, 4, 1, cf) == 1) &&
                             (fread(&file_key_count, 8, 1, cf) == 1);
            
            if (header_ok && magic == 0x424C4F4F4D465430ULL /* "BLOOMFT0" */ &&
                version == 1 &&
                file_num_bits == bloom_num_bits &&
                (int)file_num_hashes == bloom_num_hashes) {
                
                printf("[Bloom Cache] 找到快取檔案，參數匹配 (bits=%llu, hashes=%d, keys=%llu)\n",
                       (unsigned long long)file_num_bits, (int)file_num_hashes,
                       (unsigned long long)file_key_count);
                printf("[Bloom Cache] 正在從檔案載入 %.2f MB...\n",
                       (double)bloom_array_size * sizeof(uint64_t) / 1024.0 / 1024.0);
                
                // 分批讀入 host buffer 再上傳 GPU（避免一次分配太大 host memory）
                const size_t chunk_words = 16 * 1024 * 1024; // 128MB per chunk
                uint64_t *host_buf = (uint64_t *)malloc(chunk_words * sizeof(uint64_t));
                if (host_buf) {
                    size_t remaining = bloom_array_size;
                    size_t offset = 0;
                    bool read_ok = true;
                    while (remaining > 0) {
                        size_t to_read = (remaining < chunk_words) ? remaining : chunk_words;
                        if (fread(host_buf, sizeof(uint64_t), to_read, cf) != to_read) {
                            fprintf(stderr, "[Bloom Cache] 讀取失敗，將從 Redis 重建\n");
                            read_ok = false;
                            break;
                        }
                        checkCuda(cudaMemcpy(dev_bloom_bits + offset, host_buf,
                                            to_read * sizeof(uint64_t), cudaMemcpyHostToDevice),
                                 "Bloom cache upload");
                        offset += to_read;
                        remaining -= to_read;
                    }
                    free(host_buf);
                    if (read_ok) {
                        loaded_from_cache = true;
                        bloom_loaded_count = file_key_count;
                        bloom_filter_ready = true;
                        bloom_loader_done = true;
                        printf("[Bloom Cache] 載入完成！%llu 個 keys，Bloom 已啟用\n",
                               (unsigned long long)file_key_count);
                    }
                }
            } else if (header_ok) {
                printf("[Bloom Cache] 快取檔案參數不匹配 (file: bits=%llu hashes=%d, need: bits=%llu hashes=%d)，將重建\n",
                       (unsigned long long)file_num_bits, (int)file_num_hashes,
                       (unsigned long long)bloom_num_bits, bloom_num_hashes);
            }
            fclose(cf);
        } else {
            printf("[Bloom Cache] 無快取檔案 (%s)，將從 Redis 建立\n", bloom_cache_path);
        }
    }
    
    // 若快取未命中，從 Redis SSCAN 建立，完成後存檔
    if (!loaded_from_cache) {
        std::thread bloom_loader([&]() {
            redisContext *rc = redisConnect("127.0.0.1", 6379);
            if (rc == nullptr || rc->err) {
                fprintf(stderr, "[Bloom Loader] Redis連線失敗: %s\n", (rc && rc->errstr) ? rc->errstr : "unknown");
                if (rc) redisFree(rc);
                bloom_loader_done = true;
                return;
            }

            timeval redis_timeout;
            redis_timeout.tv_sec = 30;
            redis_timeout.tv_usec = 0;
            redisSetTimeout(rc, redis_timeout);

            printf("[Bloom Loader] 開始從 Redis SSCAN 載入 Bloom bits...\n");
            
            uint64_t cursor = 0;
            const uint64_t batch_size = 1000000;
            size_t loader_capacity = (size_t)batch_size;
            char *batch_hex = new char[loader_capacity * 11];
            char *dev_loader_hex_buffer = nullptr;
            checkCuda(cudaMalloc(&dev_loader_hex_buffer, loader_capacity * 11), "Bloom loader hex malloc");
            cudaStream_t bloom_stream;
            checkCuda(cudaStreamCreate(&bloom_stream), "Bloom loader stream create");
            bool load_ok = true;
            int consecutive_errors = 0;
            
            do {
                char cmd[256];
                snprintf(cmd, sizeof(cmd), "SSCAN mongo_keys %llu COUNT %llu",
                         (unsigned long long)cursor,
                         (unsigned long long)batch_size);
                redisReply *reply = (redisReply *)redisCommand(rc, cmd);
                
                if (reply == nullptr || reply->type != REDIS_REPLY_ARRAY || reply->elements != 2) {
                    consecutive_errors++;
                    fprintf(stderr, "[Bloom Loader] SSCAN 失敗 (retry=%d): %s\n", consecutive_errors,
                            rc->errstr ? rc->errstr : "unexpected reply");
                    if (reply) freeReplyObject(reply);
                    if (consecutive_errors >= 5) {
                        load_ok = false;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200 * consecutive_errors));
                    continue;
                }
                consecutive_errors = 0;
                
                redisReply *cursor_reply = reply->element[0];
                if (cursor_reply->type == REDIS_REPLY_STRING) {
                    cursor = strtoull(cursor_reply->str, nullptr, 10);
                }
                
                redisReply *keys_reply = reply->element[1];
                if (keys_reply->type == REDIS_REPLY_ARRAY) {
                    if (keys_reply->elements > loader_capacity) {
                        size_t new_capacity = keys_reply->elements;
                        delete[] batch_hex;
                        batch_hex = new char[new_capacity * 11];
                        if (dev_loader_hex_buffer) cudaFree(dev_loader_hex_buffer);
                        checkCuda(cudaMalloc(&dev_loader_hex_buffer, new_capacity * 11), "Bloom loader hex realloc");
                        loader_capacity = new_capacity;
                        printf("\n[Bloom Loader] 緩衝區擴容至 %zu keys\n", loader_capacity);
                    }

                    size_t valid_count = 0;
                    for (size_t i = 0; i < keys_reply->elements; i++) {
                        const char *key = keys_reply->element[i]->str;
                        if (key != nullptr && strlen(key) >= 11) {
                            memcpy(batch_hex + valid_count * 11, key, 11);
                            valid_count++;
                        }
                    }
                    
                    if (valid_count > 0) {
                        size_t copy_size = valid_count * 11;
                        checkCuda(cudaMemcpyAsync(dev_loader_hex_buffer, batch_hex, copy_size, cudaMemcpyHostToDevice, bloom_stream),
                                 "Bloom loader memcpy");
                        
                        int threads = 256;
                        int blocks = ((int)valid_count + threads - 1) / threads;
                        gpu_bloom_add_batch<<<blocks, threads, 0, bloom_stream>>>(dev_bloom_bits, bloom_num_bits, bloom_num_hashes,
                                                                                   dev_loader_hex_buffer, (int)valid_count, 11);
                        
                        checkCuda(cudaStreamSynchronize(bloom_stream), "Bloom add batch");
                        bloom_loaded_count += valid_count;
                        
                        printf("[Bloom Loader] \r已加載 %llu 個 keys...",
                            (unsigned long long)bloom_loaded_count.load());
                        fflush(stdout);
                    }
                }
                
                freeReplyObject(reply);
                
            } while (cursor != 0);
            
            if (dev_loader_hex_buffer) cudaFree(dev_loader_hex_buffer);
            cudaStreamDestroy(bloom_stream);
            delete[] batch_hex;
            redisFree(rc);
            bloom_filter_ready = (load_ok && cursor == 0);
            bloom_loader_done = true;
            
            if (bloom_filter_ready.load()) {
                printf("\n[Bloom Loader] 完成！已加載 %llu 個 keys\n",
                       (unsigned long long)bloom_loaded_count.load());
                
                // === 存檔 Bloom 快取到磁碟 ===
                printf("[Bloom Cache] 正在存檔 %.2f MB 到 %s...\n",
                       (double)bloom_array_size * sizeof(uint64_t) / 1024.0 / 1024.0,
                       bloom_cache_path);
                FILE *cf = fopen(bloom_cache_path, "wb");
                if (cf) {
                    uint64_t magic = 0x424C4F4F4D465430ULL; // "BLOOMFT0"
                    uint32_t version = 1;
                    uint64_t file_num_bits = bloom_num_bits;
                    uint32_t file_num_hashes = (uint32_t)bloom_num_hashes;
                    uint64_t file_key_count = bloom_loaded_count.load();
                    fwrite(&magic, 8, 1, cf);
                    fwrite(&version, 4, 1, cf);
                    fwrite(&file_num_bits, 8, 1, cf);
                    fwrite(&file_num_hashes, 4, 1, cf);
                    fwrite(&file_key_count, 8, 1, cf);
                    
                    const size_t chunk_words = 16 * 1024 * 1024; // 128MB per chunk
                    uint64_t *host_buf = (uint64_t *)malloc(chunk_words * sizeof(uint64_t));
                    if (host_buf) {
                        size_t remaining = bloom_array_size;
                        size_t offset = 0;
                        while (remaining > 0) {
                            size_t to_write = (remaining < chunk_words) ? remaining : chunk_words;
                            checkCuda(cudaMemcpy(host_buf, dev_bloom_bits + offset,
                                                to_write * sizeof(uint64_t), cudaMemcpyDeviceToHost),
                                     "Bloom cache download");
                            fwrite(host_buf, sizeof(uint64_t), to_write, cf);
                            offset += to_write;
                            remaining -= to_write;
                        }
                        free(host_buf);
                    }
                    fclose(cf);
                    printf("[Bloom Cache] 存檔完成！下次啟動將自動載入\n");
                } else {
                    fprintf(stderr, "[Bloom Cache] 無法建立快取檔案: %s\n", bloom_cache_path);
                }
            } else {
                printf("\n[Bloom Loader] 未完成（已加載 %llu 個 keys），維持 Redis 全量校驗\n",
                       (unsigned long long)bloom_loaded_count.load());
            }
        });
        printf("[Bloom Loader] 等待 Bloom 載入完成後再啟動主流程...\n");
        bloom_loader.join();
    }
    
    if (!bloom_filter_ready.load()) {
        fprintf(stderr, "[Bloom] 初始化失敗，終止主流程以避免漏檢。\n");
        mongoc_cleanup();
        return 1;
    }
    printf("[Bloom] 就緒，開始主流程。\n");
    
    SafeQueue<BatchTask> task_queue;
    SafeQueue<int> free_indices;
    for (int i = 0; i < num_streams; i++) free_indices.push(i);

    std::thread mongo_logger(MongoMatchLogger, "mongodb://127.0.0.1:27017", "127.0.0.1", 6379);

    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; i++) {
        workers.emplace_back([&task_queue, &free_indices, streams, dev_bloom_bits, bloom_num_bits, bloom_num_hashes, &bloom_filter_ready]() {
            HybridScanner scanner("127.0.0.1", 6379, dev_bloom_bits, bloom_num_bits, bloom_num_hashes, &bloom_filter_ready);
            while (true) {
                BatchTask task = task_queue.pop();
                if (task.n == -1) break; // Sentinel to exit
                
                checkCuda(cudaStreamSynchronize(streams[task.stream_idx]), "sync in worker");
                
                scanner.checkMatchBatch(task.k_start, task.r_ptr, task.n);
                free_indices.push(task.stream_idx);
            }
        });
    }

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    printf("Starting Producer loop (GPU) with %d Workers...\n", num_workers);

    std::atomic<long long> total_keys(0);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::atomic<bool> stats_stop(false);
    std::thread stats_reporter([&]() {
        while (!stats_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            if (stats_stop.load(std::memory_order_relaxed)) break;

            unsigned long long total_scanned = g_stat_total_scanned.load(std::memory_order_relaxed);
            unsigned long long bloom_filtered = g_stat_bloom_filtered.load(std::memory_order_relaxed);
            unsigned long long redis_queried = g_stat_redis_queried.load(std::memory_order_relaxed);
            bool bloom_ready = bloom_filter_ready.load(std::memory_order_relaxed);
            bool loader_done = bloom_loader_done.load(std::memory_order_relaxed);
            unsigned long long loaded = (unsigned long long)bloom_loaded_count.load(std::memory_order_relaxed);

            double bloom_filtered_pct = (total_scanned > 0)
                ? (100.0 * (double)bloom_filtered / (double)total_scanned)
                : 0.0;
            double redis_queried_pct = (total_scanned > 0)
                ? (100.0 * (double)redis_queried / (double)total_scanned)
                : 0.0;

                     printf("\n[STATS] scanned=%llu | bloom_filtered=%llu (%.2f%%) | redis_queried=%llu (%.2f%%) | loader_done=%d | bloom_ready=%d | loaded=%llu\n",
                   total_scanned,
                   bloom_filtered,
                   bloom_filtered_pct,
                   redis_queried,
                         redis_queried_pct,
                         loader_done ? 1 : 0,
                         bloom_ready ? 1 : 0,
                         loaded);
            fflush(stdout);
        }
    });

    do {
        // 從空閒索引隊列取出一個 Buffer
        int idx = free_indices.pop();

        // 1. CPU 產生隨機 k_start
        uint64_t k_start[4];
        generate_random_k(k_start);

        // 2. 啟動 GPU 運算 (k = k_start + tid)
        ComputeKG_Jacobian<<<blocks, threads, 0, streams[idx]>>> (
            k_start[0], k_start[1], k_start[2], k_start[3], 
            d_X[idx], d_Y[idx], d_Z[idx], n
        );
        ParallelConvertToR<<<blocks, threads, 0, streams[idx]>>> (
            d_X[idx], d_Z[idx], d_r[idx], n
        );
        
        // 3. 非同步拷貝回 Host
        checkCuda(cudaMemcpyAsync(h_r[idx], d_r[idx], X_bytes, cudaMemcpyDeviceToHost, streams[idx]), "async cpy");

        // 4. 立即丟進 Task Queue
        BatchTask task;
        task.stream_idx = idx;
        memcpy(task.k_start, k_start, 32);
        task.r_ptr = h_r[idx];
        task.n = n;
        task_queue.push(task);

        total_keys += n;
        static auto last_print = start_time;
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed = current_time - start_time;
        std::chrono::duration<double> print_elapsed = current_time - last_print;

        if (print_elapsed.count() > 2.0) {
            double speed = total_keys / total_elapsed.count();
            printf("\r[PRODUCER] Speed: %.2f keys/sec | Total: %lld | FreeBuff: %d", 
                   speed, (long long)total_keys, (int)num_streams - (int)task_queue.size()); // Approx check
            fflush(stdout);
            last_print = current_time;
        }

    } while (infinite);

    // Shutdown workers
    for (int i = 0; i < num_workers; i++) {
        BatchTask sentinel;
        sentinel.n = -1;
        task_queue.push(sentinel);
    }
    for (auto &t : workers) t.join();

    stats_stop.store(true, std::memory_order_relaxed);
    if (stats_reporter.joinable()) stats_reporter.join();

    // Shutdown mongo logger
    MatchData m_sentinel;
    m_sentinel.exit_signal = true;
    g_match_queue.push(m_sentinel);
    mongo_logger.join();

    // Cleanup MongoDB Driver (Global)
    mongoc_cleanup();

    return 0;
}
