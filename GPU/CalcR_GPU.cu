/*
 * CalcR_GPU.cu
 *
 * CPU 產生 k 值 → GPU 計算 r = x(k·G) mod n (secp256k1 ECDSA)
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
 *   ./CalcR_GPU [count] [k_start]  # count: 要計算的 k 值數量，預設 16
 *                                  # k_start: 起始 k 值，預設 1（可為大整數）
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
// 256-bit double-and-add (Jacobian): 計算 k*G, 結果 Jacobian 座標
// k 以 4 個 uint64 (小端) 表示。
// ===========================================================================
__device__ void ScalarMultG_Jacobian(
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4],
    const uint64_t k[4])
{
    // 初始化結果為無窮遠點 (用 Z=0 表示)
    Rx[0]=0; Rx[1]=0; Rx[2]=0; Rx[3]=0;
    Ry[0]=1; Ry[1]=0; Ry[2]=0; Ry[3]=0;
    Rz[0]=0; Rz[1]=0; Rz[2]=0; Rz[3]=0;  // Z=0 → 無窮遠點

    uint64_t Tx[4], Ty[4], Tz[4]; // 暫存倍增點

    // 從最高有效位元開始，逐位元 double-and-add
    bool started = false;

    for (int word = 3; word >= 0; word--) {
        for (int bit = 63; bit >= 0; bit--) {
            if (started) {
                // 倍增
                JacDouble(Tx, Ty, Tz, Rx, Ry, Rz);
                Rx[0]=Tx[0]; Rx[1]=Tx[1]; Rx[2]=Tx[2]; Rx[3]=Tx[3];
                Ry[0]=Ty[0]; Ry[1]=Ty[1]; Ry[2]=Ty[2]; Ry[3]=Ty[3];
                Rz[0]=Tz[0]; Rz[1]=Tz[1]; Rz[2]=Tz[2]; Rz[3]=Tz[3];
            }
            if ((k[word] >> bit) & 1ULL) {
                if (!started) {
                    // 第一個 1 bit：直接把 G 放入結果
                    Rx[0]=DEV_GX[0]; Rx[1]=DEV_GX[1]; Rx[2]=DEV_GX[2]; Rx[3]=DEV_GX[3];
                    Ry[0]=DEV_GY[0]; Ry[1]=DEV_GY[1]; Ry[2]=DEV_GY[2]; Ry[3]=DEV_GY[3];
                    Rz[0]=1;         Rz[1]=0;          Rz[2]=0;          Rz[3]=0; // Z=1
                    started = true;
                } else {
                    // 一般加法：Jacobian + Affine(G)
                    JacMixedAdd(Tx, Ty, Tz, Rx, Ry, Rz, DEV_GX, DEV_GY);
                    Rx[0]=Tx[0]; Rx[1]=Tx[1]; Rx[2]=Tx[2]; Rx[3]=Tx[3];
                    Ry[0]=Ty[0]; Ry[1]=Ty[1]; Ry[2]=Ty[2]; Ry[3]=Ty[3];
                    Rz[0]=Tz[0]; Rz[1]=Tz[1]; Rz[2]=Tz[2]; Rz[3]=Tz[3];
                }
            }
        }
    }
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

static void to_hex48(char *dest, uint64_t v0) {
    uint64_t val = v0 & 0xFFFFFFFFFFFFULL;
    for (int j = 11; j >= 0; j--) {
        dest[j] = HEX_CHARS[val & 0xf];
        val >>= 4;
    }
    dest[12] = '\0';
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

// ===========================================================================
// Hybrid Scanner Logic (Redis + MongoDB)
// ===========================================================================
class HybridScanner {
public:
    redisContext *redis_ctx = nullptr;
    mongoc_client_t *mongo_client = nullptr;
    mongoc_collection_t *match_col = nullptr;
    mongoc_collection_t *btc_col = nullptr;
    char* tag_buffer = nullptr;
    int current_n = 0;

    HybridScanner(const char *redis_ip, int redis_port, const char *mongo_uri) {
        // Connect to Redis
        redis_ctx = redisConnect(redis_ip, redis_port);
        if (redis_ctx == nullptr || redis_ctx->err) {
            if (redis_ctx) printf("Redis Connection Error: %s\n", redis_ctx->errstr);
            else printf("Redis Connection Error: Can't allocate redis context\n");
            exit(1);
        }
        redisReply *reply = (redisReply *)redisCommand(redis_ctx, "SELECT 0");
        freeReplyObject(reply);

        // Connect to MongoDB (mongoc_init() must be called once in main)
        mongo_client = mongoc_client_new(mongo_uri);
        if (!mongo_client) {
            printf("MongoDB Connection Error: Failed to parse URI\n");
            exit(1);
        }
        match_col = mongoc_client_get_collection(mongo_client, "ecdsa", "match");
        btc_col = mongoc_client_get_collection(mongo_client, "ecdsa", "btc");
    }

    ~HybridScanner() {
        if (redis_ctx) redisFree(redis_ctx);
        if (tag_buffer) delete[] tag_buffer;

        if (match_col) mongoc_collection_destroy(match_col);
        if (btc_col) mongoc_collection_destroy(btc_col);
        if (mongo_client) mongoc_client_destroy(mongo_client);
    }

    void checkMatchBatch(const uint64_t k_start[4], const uint64_t *rs_ptr, int n) {
        if (n <= 0) return;

        const int sub_batch_size = 50000;
        
        if (sub_batch_size > current_n) {
            if (tag_buffer) delete[] tag_buffer;
            tag_buffer = new char[sub_batch_size * 13];
            current_n = sub_batch_size;
        }

        for (int start = 0; start < n; start += sub_batch_size) {
            int current_batch = (n - start < sub_batch_size) ? (n - start) : sub_batch_size;

            std::vector<const char*> argv;
            std::vector<size_t> argvlen;

            argv.push_back("SMISMEMBER");
            argvlen.push_back(10);
            argv.push_back("mongo_keys");
            argvlen.push_back(10);

            for (int i = 0; i < current_batch; i++) {
                to_hex48(tag_buffer + i * 13, rs_ptr[(start + i) * 4]);
                argv.push_back(tag_buffer + i * 13);
                argvlen.push_back(12);
            }

            redisReply *reply = (redisReply *)redisCommandArgv(redis_ctx, argv.size(), argv.data(), argvlen.data());
            
            if (reply == nullptr) {
                fprintf(stderr, "Redis error: %s\n", redis_ctx->errstr);
                continue;
            }

            if (reply->type == REDIS_REPLY_ARRAY) {
                for (size_t j = 0; j < reply->elements; j++) {
                    if (reply->element[j]->integer == 1) {
                        // Redis 命中！還原 k 值
                        uint64_t k_match[4];
                        uint64_t carry = (uint64_t)(start + j);
                        for (int limb = 0; limb < 4; limb++) {
                            k_match[limb] = k_start[limb] + carry;
                            carry = (k_match[limb] < k_start[limb]) ? 1 : 0;
                        }

                        std::string ks = hex256(k_match);
                        std::string rs = hex256(rs_ptr + (start + j) * 4);
                        
                        printf("\n[MATCH FOUND!] k = %s\n", ks.c_str());
                        printf("               r = %s\n", rs.c_str());

                        // 1. 寫入 MongoDB 的 match 集合 (r值跟k值)
                        bson_t *doc = bson_new();
                        BSON_APPEND_UTF8(doc, "r", rs.c_str());
                        BSON_APPEND_UTF8(doc, "k", ks.c_str());
                        
                        bson_error_t error;
                        if (!mongoc_collection_insert_one(match_col, doc, NULL, NULL, &error)) {
                            fprintf(stderr, "MongoDB Insert Error: %s\n", error.message);
                        }

                        // 2. 拿 r 值，連到 btc 集合查詢 id 欄位
                        // 同時查詢: r, 0+r, 00+r
                        std::string r0 = "0" + rs;
                        std::string r00 = "00" + rs;

                        // 使用 $in 查詢: { "id": { "$in": [rs, r0, r00] } }
                        bson_t *query = bson_new();
                        bson_t in_child;
                        BSON_APPEND_DOCUMENT_BEGIN(query, "id", &in_child);
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
                            // 3. 若符合，更新 match 資料為符合 (y值) 並寫下時間
                            bson_t *selector = bson_new();
                            BSON_APPEND_UTF8(selector, "r", rs.c_str());
                            BSON_APPEND_UTF8(selector, "k", ks.c_str());

                            bson_t *update = bson_new();
                            bson_t child;
                            BSON_APPEND_DOCUMENT_BEGIN(update, "$set", &child);
                            BSON_APPEND_UTF8(&child, "found", "y");
                            BSON_APPEND_TIME_T(&child, "time", time(NULL));
                            bson_append_document_end(update, &child);

                            if (!mongoc_collection_update_one(match_col, selector, update, NULL, NULL, &error)) {
                                fprintf(stderr, "MongoDB Update Error: %s\n", error.message);
                            }
                            bson_destroy(selector);
                            bson_destroy(update);
                            printf("               [PRECISE MATCH!] Updated with found='y'\n");
                        }
                        
                        mongoc_cursor_destroy(cursor);
                        bson_destroy(query);
                        bson_destroy(doc);
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

    uint64_t k_start[4] = {1, 0, 0, 0};
    if (argc > 2) parse256(k_start, argv[2]);
    if (k_start[0] == 0 && k_start[1] == 0 && k_start[2] == 0 && k_start[3] == 0) k_start[0] = 1;

    printf("=== CalcR_GPU: secp256k1 r = x(k*G) mod n (Multi-threaded Extreme) ===\n");
    printf("批次大小: %d\n", n);
    printHex256("起始 k", k_start);
    
    // Initialize MongoDB Driver (Global)
    mongoc_init();

    checkCuda(cudaMemcpyToSymbol(DEV_P,  HOST_P,  32), "cpyP");
    checkCuda(cudaMemcpyToSymbol(DEV_N,  HOST_N,  32), "cpyN");
    checkCuda(cudaMemcpyToSymbol(DEV_GX, HOST_GX, 32), "cpyGx");
    checkCuda(cudaMemcpyToSymbol(DEV_GY, HOST_GY, 32), "cpyGy");

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
    SafeQueue<BatchTask> task_queue;
    SafeQueue<int> free_indices;
    for (int i = 0; i < num_streams; i++) free_indices.push(i);

    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; i++) {
        workers.emplace_back([&task_queue, &free_indices, streams]() {
            HybridScanner scanner("host.docker.internal", 6379, "mongodb://host.docker.internal:27017");
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

    do {
        // 從空閒索引隊列取出一個 Buffer
        int idx = free_indices.pop();

        // 2. 啟動 GPU 運算 (直接傳入 k_start，不需要在 CPU 算 1M 次)
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

        // 遞增 k_start
        uint64_t carry = (uint64_t)n;
        for (int j = 0; j < 4; j++) {
            uint64_t val = k_start[j];
            k_start[j] = val + carry;
            carry = (k_start[j] < val) ? 1 : 0;
        }

    } while (infinite);

    // Shutdown workers
    for (int i = 0; i < num_workers; i++) {
        BatchTask sentinel;
        sentinel.n = -1;
        task_queue.push(sentinel);
    }
    for (auto &t : workers) t.join();

    // Cleanup MongoDB Driver (Global)
    mongoc_cleanup();

    return 0;
}
