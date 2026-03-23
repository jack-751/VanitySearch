/*
 * CalcR_GPU.cu
 *
 * CPU 產生 k 值，GPU 計算 secp256k1 的 r = x(k·G) mod n。
 *
 * 策略：
 *   ① 每個 CUDA thread 負責一個 k 值。
 *   ② scalar multiplication 採用 Jacobian 座標下的 double-and-add，
 *      讓點加法與倍點全程避免每步做 modular inversion。
 *   ③ 所有模數運算複用 VanitySearch 的 GPUMath.h 核心
 *      (_ModMult, _ModSqr, _ModInv)。
 *   ④ 最後由 Jacobian Z 還原 affine x 時，現階段採用 per-thread
 *      inversion 的正確性優先版本；若後續要再追吞吐量，再重做
 *      正確的 batch inversion 版本。
 *
 * Jacobian 座標 (X:Y:Z) 對應仿射 (x, y) = (X/Z^2, Y/Z^3)
 *   點加法 / 點倍增全程無需 ModInv。
 *   最後 Z != 0 時，x_affine = X * Z^{-2} mod P。
 *
 * 編譯：
 *   nvcc -O2 -arch=sm_86 -I. CalcR_GPU.cu -o CalcR_GPU $(pkg-config --cflags --libs libmongoc-1.0)
 *
 * 執行：
 *   ./CalcR_GPU [count] [k_start]
 *
 * 已驗證測試向量：
 *   k = 22860751503568827944108675187057424959908371263446804931816731781078483855304
 * (22860751503568827944108675187057424959908371263446804931816731781078483855304)10 = (0X328ABA10DD1E344A132F2818677E2D0318E14A7B6A307F426A94F8114701E7C8)16
 *   r = 0xc0199ab7191cd18ccea7e4e9a4fd8ee4a970b7cbb48b8077190f5f69ebfe57d2
 * 
 * MONGO_URI=mongodb://127.0.0.1:27017 MONGO_DB=ecdsa MONGO_CANDIDATE_COLLECTION=matched_candidates MONGO_KEYS_CACHE_FILE=mongo_keys_16.cache.bin CACHE_HEX_LEN=16 SCAN_WORKERS=1 MAX_PENDING_SCAN=8 MAX_PENDING_MATCH=20000 ./CalcR_GPU 0 1
 * 
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
#include <memory>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <random>
#include <mongoc/mongoc.h>
#include <bson/bson.h>
#include <ctime>
#include "BlockedBloom.h"

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

 
     已驗證測試向量：
         k = 22860751503568827944108675187057424959908371263446804931816731781078483855304
         r = 0xc0199ab7191cd18ccea7e4e9a4fd8ee4a970b7cbb48b8077190f5f69ebfe57d2
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
// Jacobian Z 轉回 affine x，再對 n 取模得 r
// 輸入：Zs[n][5] (5-limb，Z 在 Jacobian 座標)
//       Xs[n][4]
// 輸出：rs[n][4] = (X/Z^2) mod n
//
// 目前採用每個 thread 各自做一次 ModInv 的正確性優先版本。
// 若後續需要更高吞吐量，可在確保正確性的前提下重新引入 batch inverse。
// ===========================================================================
__global__ void ParallelConvertToR(
    uint64_t *Xs,          // [n * 4]  Jacobian X
    uint64_t *Zs,          // [n * 5]  Jacobian Z (5 limbs)
    uint64_t *rs,          // [n * 4]  output r values
    int       n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Jacobian Z=0 means point-at-infinity; keep r=0.
    if (Zs[tid*5+0]==0 && Zs[tid*5+1]==0 && Zs[tid*5+2]==0 && Zs[tid*5+3]==0) {
        rs[tid*4+0] = rs[tid*4+1] = rs[tid*4+2] = rs[tid*4+3] = 0;
        return;
    }

    uint64_t invZ5[5] = {
        Zs[tid*5+0], Zs[tid*5+1], Zs[tid*5+2], Zs[tid*5+3], 0
    };
    _ModInv(invZ5);

    uint64_t invZ2[5];
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

static std::string hex256(const uint64_t v[4]) {
    char buf[65];
    to_hex256(buf, v);
    return std::string(buf);
}

static uint64_t read_le_u64_n(const uint8_t *p, int nbytes) {
    uint64_t v = 0;
    for (int i = 0; i < nbytes; i++) v |= ((uint64_t)p[i]) << (8 * i);
    return v;
}

static void write_le_u64_n(uint8_t *p, int nbytes, uint64_t v) {
    for (int i = 0; i < nbytes; i++) p[i] = (uint8_t)((v >> (8 * i)) & 0xFFULL);
}

static bool parse_cache_bits(const char magic[8], int *bits_out) {
    if (!magic || !bits_out) return false;
    if (!(magic[0] == 'B' && magic[1] == 'T' && magic[2] == 'C' && magic[3] == 'T' && magic[4] == 'A')) {
        return false;
    }
    if (magic[5] < '0' || magic[5] > '9' || magic[6] < '0' || magic[6] > '9' || magic[7] < '0' || magic[7] > '9') {
        return false;
    }
    int bits = (magic[5] - '0') * 100 + (magic[6] - '0') * 10 + (magic[7] - '0');
    if (!(bits == 48 || bits == 56 || bits == 64)) return false;
    *bits_out = bits;
    return true;
}

static bool parse_last_hex_u64(uint64_t *out, const char *s, uint32_t len, int hex_len) {
    if (!out || !s || hex_len <= 0 || len < (uint32_t)hex_len) return false;
    const char *p = s + len - hex_len;
    uint64_t v = 0;
    for (int i = 0; i < hex_len; i++) {
        char c = p[i];
        uint64_t nibble = 0;
        if (c >= '0' && c <= '9') nibble = (uint64_t)(c - '0');
        else if (c >= 'a' && c <= 'f') nibble = (uint64_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') nibble = (uint64_t)(c - 'A' + 10);
        else return false;
        v = (v << 4) | nibble;
    }
    *out = v;
    return true;
}

static inline void extract_tail_le_bytes_from_r(uint8_t *dst, int hex_len, const uint64_t *r_limbs) {
    int nbytes = (hex_len + 1) / 2;
    uint64_t lo = r_limbs[0];
    uint64_t hi = r_limbs[1];

    for (int i = 0; i < nbytes; i++) {
        if (i < 8) dst[i] = (uint8_t)((lo >> (8 * i)) & 0xFFULL);
        else dst[i] = (uint8_t)((hi >> (8 * (i - 8))) & 0xFFULL);
    }
    if (hex_len & 1) {
        dst[nbytes - 1] &= 0x0F;
    }
}

class MongoKeyCache {
public:
    MongoKeyCache() {
        setTagBits(48);
    }

    bool configureHexLen(int hex_len) {
        if (!(hex_len == 12 || hex_len == 14 || hex_len == 16)) return false;
        setTagBits(hex_len * 4);
        return true;
    }

    int tagHexLen() const { return tag_hex_len; }
    int tagBits() const { return tag_bits; }
    uint64_t tagMask() const { return tag_mask; }

    bool loadFromFile(const char *file_path) {
        if (!file_path || file_path[0] == '\0') return false;

        printf("Trying local cache file: %s\n", file_path);
        fflush(stdout);

        std::ifstream in(file_path, std::ios::binary);
        if (!in.is_open()) {
            fprintf(stderr, "MongoKeyCache: cannot open cache file: %s\n", file_path);
            return false;
        }

        in.seekg(0, std::ios::end);
        std::streamoff file_size = in.tellg();
        in.seekg(0, std::ios::beg);

        char magic[8] = {0};
        uint64_t count = 0;
        in.read(magic, sizeof(magic));
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (!in.good()) {
            fprintf(stderr, "MongoKeyCache: failed to read cache header from: %s\n", file_path);
            return false;
        }

        int file_bits = 0;
        if (!parse_cache_bits(magic, &file_bits)) {
            fprintf(stderr,
                "MongoKeyCache: unsupported cache magic in %s (got=%.8s expected=BTCTA048/056/064)\n",
                    file_path, magic);
            return false;
        }
        setTagBits(file_bits);

        const uint64_t expected_payload = count * (uint64_t)entry_bytes;
        const uint64_t expected_total = expected_payload + 16ULL;
        if (file_size >= 0 && (uint64_t)file_size != expected_total) {
            fprintf(stderr,
                    "MongoKeyCache: cache size mismatch in %s (actual=%llu expected=%llu)\n",
                    file_path,
                    (unsigned long long)file_size,
                    (unsigned long long)expected_total);
            return false;
        }

        printf("Cache header OK: entries=%llu, payload=%.2f GiB\n",
               (unsigned long long)count,
               (double)expected_payload / (1024.0 * 1024.0 * 1024.0));
        printf("Reading cache payload...\n");
        fflush(stdout);

        try {
            values.resize((size_t)count);
        } catch (const std::bad_alloc &) {
            fprintf(stderr,
                    "MongoKeyCache: not enough RAM to allocate %llu entries from cache file: %s\n",
                    (unsigned long long)count,
                    file_path);
            values.clear();
            return false;
        }

        if (count > 0) {
            std::vector<uint8_t> payload((size_t)expected_payload);
            in.read(reinterpret_cast<char*>(payload.data()), (std::streamsize)payload.size());
            if (!in.good()) {
                fprintf(stderr, "MongoKeyCache: failed while reading cache payload from: %s\n", file_path);
                values.clear();
                return false;
            }
            for (uint64_t i = 0; i < count; i++) {
                values[(size_t)i] = read_le_u64_n(payload.data() + (size_t)i * (size_t)entry_bytes, entry_bytes) & tag_mask;
            }
        }

        printf("Cache payload read complete.\n");
        fflush(stdout);

        bool verify_sorted = false;
        const char *verify_env = getenv("VERIFY_CACHE_SORTED");
        if (verify_env && verify_env[0] != '\0') {
            verify_sorted = atoi(verify_env) != 0;
        }

        if (verify_sorted) {
            printf("Verifying cache sort order (VERIFY_CACHE_SORTED=1)...\n");
            fflush(stdout);
            if (!std::is_sorted(values.begin(), values.end())) {
                printf("Cache is not sorted; sorting + deduplicating...\n");
                fflush(stdout);
                std::sort(values.begin(), values.end());
                values.erase(std::unique(values.begin(), values.end()), values.end());
            }
        } else {
            printf("Skip sorted-check for fast startup (set VERIFY_CACHE_SORTED=1 to force check).\n");
        }

        printf("Loaded %zu btc tags (%d-bit) from cache file: %s\n", values.size(), tag_bits, file_path);
        return !values.empty();
    }

    bool saveToFile(const char *file_path) const {
        if (!file_path || file_path[0] == '\0' || values.empty()) return false;

        std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }

        char magic[8] = {'B','T','C','T','A','0','0','0'};
        magic[5] = (char)('0' + (tag_bits / 100));
        magic[6] = (char)('0' + ((tag_bits / 10) % 10));
        magic[7] = (char)('0' + (tag_bits % 10));
        const uint64_t count = (uint64_t)values.size();
        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));

        std::vector<uint8_t> payload((size_t)count * (size_t)entry_bytes);
        for (uint64_t i = 0; i < count; i++) {
            write_le_u64_n(payload.data() + (size_t)i * (size_t)entry_bytes, entry_bytes, values[(size_t)i]);
        }
        out.write(reinterpret_cast<const char*>(payload.data()), (std::streamsize)payload.size());
        if (!out.good()) {
            return false;
        }

        printf("Saved %zu btc tags (%d-bit) to cache file: %s\n", values.size(), tag_bits, file_path);
        return true;
    }

    bool loadFromMongo(const char *mongo_uri, const char *db_name, const char *collection_name, const char *field_name) {
        mongoc_client_t *client = mongoc_client_new(mongo_uri);
        if (!client) {
            fprintf(stderr, "MongoKeyCache: failed to parse MongoDB URI\n");
            return false;
        }

        mongoc_collection_t *col = mongoc_client_get_collection(client, db_name, collection_name);
        bson_t *query = bson_new();
        bson_t *opts = bson_new();
        BSON_APPEND_INT32(opts, "batchSize", 200000);

        mongoc_cursor_t *cursor = mongoc_collection_find_with_opts(col, query, opts, nullptr);
        const bson_t *doc = nullptr;

        values.clear();
        if (!field_name || field_name[0] == '\0') field_name = "r";
        printf("Loading btc r tags (last %d hex chars) from MongoDB into local cache...\n", tag_hex_len);
        uint64_t load_count = 0;
        while (mongoc_cursor_next(cursor, &doc)) {
            bson_iter_t it;
            const char *id_str = nullptr;
            uint32_t id_len = 0;

            if (bson_iter_init_find(&it, doc, field_name) && BSON_ITER_HOLDS_UTF8(&it)) {
                id_str = bson_iter_utf8(&it, &id_len);
            }

            if (!id_str || id_len < (uint32_t)tag_hex_len) continue;

            uint64_t tag = 0;
            if (parse_last_hex_u64(&tag, id_str, id_len, tag_hex_len)) {
                values.push_back(tag & tag_mask);
                load_count++;
                if (load_count % 1000000 == 0) {
                    printf("\r  Loaded %llu M entries...", (unsigned long long)(load_count / 1000000));
                    fflush(stdout);
                }
            }
        }

        if (mongoc_cursor_error(cursor, nullptr)) {
            fprintf(stderr, "MongoKeyCache: cursor error while loading btc tags\n");
            values.clear();
        }

        mongoc_cursor_destroy(cursor);
        bson_destroy(opts);
        bson_destroy(query);
        mongoc_collection_destroy(col);
        mongoc_client_destroy(client);

        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        printf("\rLoaded %zu btc tags (%d-bit) into local cache\n", values.size(), tag_bits);
        return !values.empty();
    }

    bool contains(uint64_t value) const {
        return std::binary_search(values.begin(), values.end(), value);
    }

    size_t containsBatch(const uint64_t *tags, uint8_t *hits, size_t n) const {
        if (!tags || !hits || n == 0 || values.empty()) return 0;

        size_t matched = 0;
        for (size_t i = 0; i < n; i++) {
            hits[i] = contains(tags[i]) ? 1 : 0;
            matched += hits[i] ? 1 : 0;
        }
        return matched;
    }

    bool empty() const {
        return values.empty();
    }

private:
    void setTagBits(int bits) {
        tag_bits = bits;
        tag_hex_len = bits / 4;
        entry_bytes = bits / 8;
        tag_mask = (bits == 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << bits) - 1ULL);
    }

    int tag_bits = 48;
    int tag_hex_len = 12;
    int entry_bytes = 6;
    uint64_t tag_mask = ((1ULL << 48) - 1ULL);
    std::vector<uint64_t> values;
};

// ===========================================================================
// Thread-Safe Task Queue for Pipelining
// ===========================================================================
struct PendingBatch {
    int stream_idx;
    uint64_t k_start[4];
    int n;
    bool exit_signal = false;
};

struct ScanTask {
    uint64_t k_start[4];
    std::shared_ptr<uint64_t> r_values;
    int n;
    bool exit_signal = false;
};

template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv;
    size_t max_size = 0; // 0 means unbounded
public:
    SafeQueue() = default;
    explicit SafeQueue(size_t cap) : max_size(cap) {}

    void push(T val) {
        std::unique_lock<std::mutex> lock(m);
        if (max_size > 0) {
            cv.wait(lock, [this]{ return q.size() < max_size; });
        }
        q.push(std::move(val));
        cv.notify_all();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this]{ return !q.empty(); });
        T val = std::move(q.front());
        q.pop();
        cv.notify_all();
        return val;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(m);
        return q.size();
    }

    void setCapacity(size_t cap) {
        {
            std::lock_guard<std::mutex> lock(m);
            max_size = cap;
        }
        cv.notify_all();
    }
};

struct MatchData {
    std::string k;
    std::string r;
    bool exit_signal = false;
};

SafeQueue<MatchData> g_match_queue;
std::atomic<long long> g_scan_candidates(0);
std::atomic<long long> g_match_enqueued(0);
std::atomic<long long> g_match_processed(0);
std::atomic<long long> g_match_precise(0);
std::atomic<long long> g_mongo_total_us(0);

void MongoMatchLogger(const char* mongo_uri, const char* mongo_db, const char* candidate_collection) {
    mongoc_client_t *client = mongoc_client_new(mongo_uri);
    if (!client) {
        fprintf(stderr, "MatchLogger: Failed to parse MongoDB URI\n");
        return;
    }
    mongoc_collection_t *candidate_col = mongoc_client_get_collection(client, mongo_db, candidate_collection);

    while (true) {
        MatchData data = g_match_queue.pop();
        if (data.exit_signal) break;

        auto t0 = std::chrono::high_resolution_clock::now();

        std::string ks = data.k;
        std::string rs = data.r;

        // 1. Upsert candidate into MongoDB matched_candidates collection.
        bson_t filter;
        bson_t update;
        bson_t opts;
        bson_error_t error;
        bson_init(&filter);
        BSON_APPEND_UTF8(&filter, "r", rs.c_str());
        BSON_APPEND_UTF8(&filter, "k", ks.c_str());

        bson_init(&update);
        bson_t set_doc;
        BSON_APPEND_DOCUMENT_BEGIN(&update, "$set", &set_doc);
        BSON_APPEND_UTF8(&set_doc, "status", "candidate");
        BSON_APPEND_TIME_T(&set_doc, "updated_at", time(NULL));
        bson_append_document_end(&update, &set_doc);

        bson_init(&opts);
        BSON_APPEND_BOOL(&opts, "upsert", true);
        if (!mongoc_collection_update_one(candidate_col, &filter, &update, &opts, nullptr, &error)) {
            fprintf(stderr, "MongoDB Candidate Upsert Error: %s\n", error.message);
        }
        bson_destroy(&opts);
        bson_destroy(&update);
        bson_destroy(&filter);

        auto t1 = std::chrono::high_resolution_clock::now();
        long long us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        g_mongo_total_us.fetch_add(us, std::memory_order_relaxed);
        g_match_processed.fetch_add(1, std::memory_order_relaxed);
    }

    mongoc_collection_destroy(candidate_col);
    mongoc_client_destroy(client);
}

// ===========================================================================
// Hybrid Scanner Logic (Local Cache + MongoDB)
// ===========================================================================
class HybridScanner {
public:
    const MongoKeyCache *local_cache = nullptr;
    const BlockedBloomFilter *bloom_filter = nullptr;
    int tag_hex_len = 12;
    std::vector<uint64_t> local_tags;
    std::vector<uint8_t> local_hits;
    std::vector<uint8_t> bloom_tags;
    bool debug_first_r = false;
    bool debug_printed = false;

    HybridScanner(const MongoKeyCache *cache, const BlockedBloomFilter *bloom, int hex_len) {
        local_cache = cache;
        bloom_filter = bloom;
        tag_hex_len = hex_len;
        const char *dbg = getenv("DEBUG_FIRST_R");
        debug_first_r = (dbg && dbg[0] != '\0' && atoi(dbg) != 0);
    }

    ~HybridScanner() {}

    void checkMatchBatch(const uint64_t k_start[4], const uint64_t *rs_ptr, int n) {
        if (n <= 0) return;

        const bool use_bloom = (bloom_filter != nullptr);
        if (!use_bloom && (!local_cache || local_cache->empty())) return;

        const size_t tag_bytes = (size_t)((tag_hex_len + 1) / 2);

        if (use_bloom) {
            if ((int)local_hits.size() < n) local_hits.resize((size_t)n);
            if (bloom_tags.size() < (size_t)n * tag_bytes) bloom_tags.resize((size_t)n * tag_bytes);

            for (int i = 0; i < n; i++) {
                uint8_t *dst = bloom_tags.data() + (size_t)i * tag_bytes;
                extract_tail_le_bytes_from_r(dst, tag_hex_len, rs_ptr + i * 4);
                local_hits[(size_t)i] = bloom_filter->possiblyContains(dst, tag_bytes) ? 1 : 0;
            }
        } else {
            if ((int)local_tags.size() < n) local_tags.resize((size_t)n);
            if ((int)local_hits.size() < n) local_hits.resize((size_t)n);

            uint64_t mask = local_cache->tagMask();

            for (int i = 0; i < n; i++) {
                local_tags[(size_t)i] = rs_ptr[i * 4] & mask;
            }
            local_cache->containsBatch(local_tags.data(), local_hits.data(), (size_t)n);

            if (debug_first_r && !debug_printed && n > 0) {
                debug_printed = true;
                printf("\n[DEBUG_FIRST_R] r0=%s tail_tag=0x%016llx hit=%d\n",
                       hex256(rs_ptr).c_str(),
                       (unsigned long long)local_tags[0],
                       (int)local_hits[0]);
                fflush(stdout);
            }
        }

        for (int i = 0; i < n; i++) {
            if (!local_hits[(size_t)i]) continue;

            g_scan_candidates.fetch_add(1, std::memory_order_relaxed);

            uint64_t k_match[4];
            uint64_t carry = (uint64_t)i;
            for (int limb = 0; limb < 4; limb++) {
                k_match[limb] = k_start[limb] + carry;
                carry = (k_match[limb] < k_start[limb]) ? 1 : 0;
            }

            MatchData mdata;
            mdata.k = hex256(k_match);
            mdata.r = hex256(rs_ptr + i * 4);
            g_match_queue.push(std::move(mdata));
            g_match_enqueued.fetch_add(1, std::memory_order_relaxed);
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

    MongoKeyCache mongo_key_cache;
    int scanner_hex_len = 12;
    const char *cache_hex_env = getenv("CACHE_HEX_LEN");
    if (cache_hex_env && cache_hex_env[0] != '\0') {
        int hex_len = atoi(cache_hex_env);
        if (hex_len < 12 || hex_len > 20) {
            fprintf(stderr, "Invalid CACHE_HEX_LEN=%s (supported: 12..20)\n", cache_hex_env);
            return 1;
        }
        scanner_hex_len = hex_len;
        if (hex_len <= 16 && !mongo_key_cache.configureHexLen(hex_len)) {
            fprintf(stderr, "Invalid CACHE_HEX_LEN=%s for exact cache mode\n", cache_hex_env);
            return 1;
        }
    }

    BlockedBloomFilter bloom_filter;
    BlockedBloomFilter *bloom_ptr = nullptr;
    const char *bloom_file_env = getenv("MONGO_BLOOM_FILE");
    if (bloom_file_env && bloom_file_env[0] != '\0') {
        if (!bloom_filter.load(bloom_file_env)) {
            fprintf(stderr, "Failed to load bloom file: %s\n", bloom_file_env);
            return 1;
        }
        if ((int)bloom_filter.hexLen() != scanner_hex_len) {
            fprintf(stderr, "Bloom hex-len mismatch: bloom=%u CACHE_HEX_LEN=%d\n", bloom_filter.hexLen(), scanner_hex_len);
            return 1;
        }
        bloom_ptr = &bloom_filter;
        printf("MONGO_BLOOM_FILE=%s (k=%u blocks=%llu)\n",
               bloom_file_env,
               bloom_filter.kHashes(),
               (unsigned long long)bloom_filter.numBlocks());
    }

    const char *cache_file_env = getenv("MONGO_KEYS_CACHE_FILE");
    const char *cache_file = (cache_file_env && cache_file_env[0]) ? cache_file_env : "mongo_keys.cache.bin";
    printf("MONGO_KEYS_CACHE_FILE=%s\n", cache_file);
    printf("CACHE_HEX_LEN=%d\n", scanner_hex_len);
    fflush(stdout);

    bool local_cache_ready = false;
    if (!bloom_ptr) {
        local_cache_ready = mongo_key_cache.loadFromFile(cache_file);
        if (!local_cache_ready) {
            if (scanner_hex_len > 16) {
                fprintf(stderr, "Exact cache fallback only supports up to 16 hex; set MONGO_BLOOM_FILE for >16.\n");
                return 1;
            }

            printf("Local cache load failed; fallback to MongoDB scan...\n");
            fflush(stdout);
            const char *mongo_uri = getenv("MONGO_URI");
            const char *mongo_db = getenv("MONGO_DB");
            const char *mongo_collection = getenv("MONGO_COLLECTION");
            const char *mongo_field = getenv("MONGO_FIELD");
            if (!mongo_uri || mongo_uri[0] == '\0') mongo_uri = "mongodb://127.0.0.1:27017";
            if (!mongo_db || mongo_db[0] == '\0') mongo_db = "ecdsa";
            if (!mongo_collection || mongo_collection[0] == '\0') mongo_collection = "btc";
            if (!mongo_field || mongo_field[0] == '\0') mongo_field = "r";

            local_cache_ready = mongo_key_cache.loadFromMongo(mongo_uri, mongo_db, mongo_collection, mongo_field);
            if (local_cache_ready) {
                if (!mongo_key_cache.saveToFile(cache_file)) {
                    fprintf(stderr, "Warning: failed to save local cache file: %s\n", cache_file);
                }
            }
        }
    } else {
        local_cache_ready = true;
    }

    if (!local_cache_ready) {
        fprintf(stderr, "Failed to load local cache from file and MongoDB (ecdsa.mongo_keys).\n");
        return 1;
    }

    checkCuda(cudaMemcpyToSymbol(DEV_P,  HOST_P,  32), "cpyP");
    checkCuda(cudaMemcpyToSymbol(DEV_N,  HOST_N,  32), "cpyN");
    checkCuda(cudaMemcpyToSymbol(DEV_GX, HOST_GX, 32), "cpyGx");
    checkCuda(cudaMemcpyToSymbol(DEV_GY, HOST_GY, 32), "cpyGy");

    // -----------------------------------------------------------------------
    // 分配 Ring Buffer 與 CUDA Streams
    // -----------------------------------------------------------------------
    static const int kMaxStreams = 32;
    int num_streams = kMaxStreams;
    const char *streams_env = getenv("NUM_STREAMS");
    if (streams_env && streams_env[0] != '\0') {
        int requested = atoi(streams_env);
        if (requested > 0 && requested <= kMaxStreams) num_streams = requested;
    }

    cudaStream_t streams[kMaxStreams];
    cudaEvent_t copy_done_events[kMaxStreams];
    uint64_t *d_X[kMaxStreams], *d_Y[kMaxStreams], *d_Z[kMaxStreams], *d_r[kMaxStreams];
    uint64_t *h_r[kMaxStreams];
    for (int i = 0; i < kMaxStreams; i++) {
        streams[i] = nullptr;
        copy_done_events[i] = nullptr;
        d_X[i] = d_Y[i] = d_Z[i] = d_r[i] = nullptr;
        h_r[i] = nullptr;
    }

    size_t X_bytes = (size_t)n * 4 * sizeof(uint64_t);
    size_t Z_bytes = (size_t)n * 5 * sizeof(uint64_t);

    size_t free_mem = 0, total_mem = 0;
    checkCuda(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo");
    const size_t bytes_per_stream = X_bytes * 3 + Z_bytes; // d_X + d_Y + d_r + d_Z
    size_t usable_mem = (size_t)((double)free_mem * 0.80); // leave headroom for kernels/runtime
    int max_streams_by_mem = (bytes_per_stream > 0) ? (int)(usable_mem / bytes_per_stream) : 1;
    if (max_streams_by_mem < 1) {
        fprintf(stderr,
                "Not enough free GPU memory for even 1 stream: need %.2f GiB, free %.2f GiB (usable %.2f GiB). Reduce batch size.\n",
                (double)bytes_per_stream / (1024.0 * 1024.0 * 1024.0),
                (double)free_mem / (1024.0 * 1024.0 * 1024.0),
                (double)usable_mem / (1024.0 * 1024.0 * 1024.0));
        return 1;
    }
    if (num_streams > max_streams_by_mem) num_streams = max_streams_by_mem;
    if (num_streams > kMaxStreams) num_streams = kMaxStreams;
    if (num_streams < 1) num_streams = 1;

    printf("GPU mem free/total: %.2f / %.2f GiB\n",
           (double)free_mem / (1024.0 * 1024.0 * 1024.0),
           (double)total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("Per-stream device buffer: %.2f GiB, using streams=%d (NUM_STREAMS requested=%s)\n",
           (double)bytes_per_stream / (1024.0 * 1024.0 * 1024.0),
           num_streams,
           (streams_env && streams_env[0] != '\0') ? streams_env : "auto");

    for (int i = 0; i < num_streams; i++) {
        checkCuda(cudaStreamCreate(&streams[i]), "stream create");
        checkCuda(cudaEventCreateWithFlags(&copy_done_events[i], cudaEventDisableTiming), "event create");
        checkCuda(cudaMalloc(&d_X[i], X_bytes), "malloc X");
        checkCuda(cudaMalloc(&d_Y[i], X_bytes), "malloc Y");
        checkCuda(cudaMalloc(&d_Z[i], Z_bytes), "malloc Z");
        checkCuda(cudaMalloc(&d_r[i], X_bytes), "malloc r");
        checkCuda(cudaHostAlloc(&h_r[i], X_bytes, cudaHostAllocPortable), "hostAlloc r");
    }

    // -----------------------------------------------------------------------
    // 多執行緒掃描 Worker 池
    // 預設使用較保守的核心數，避免把 CPU 壓滿。
    // 可用 SCAN_WORKERS 覆蓋，例如：SCAN_WORKERS=4 ./CalcR_GPU 0
    // -----------------------------------------------------------------------
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 8;
    int num_workers = (int)std::max(1u, hw / 2);
    if (num_workers > 8) num_workers = 8;

    const char *workers_env = getenv("SCAN_WORKERS");
    if (workers_env && workers_env[0] != '\0') {
        int requested = atoi(workers_env);
        if (requested > 0) num_workers = requested;
    }
    SafeQueue<PendingBatch> completion_queue((size_t)num_streams * 4);
    SafeQueue<int> free_indices;
    for (int i = 0; i < num_streams; i++) free_indices.push(i);

    // Backpressure: cap pending scan batches to prevent unbounded RAM growth.
    // Approx memory per pending scan batch = X_bytes.
    int max_pending_scan = (int)((1024ULL * 1024ULL * 1024ULL) / X_bytes); // target ~1GB queue footprint
    if (max_pending_scan < 2) max_pending_scan = 2;
    if (max_pending_scan > 128) max_pending_scan = 128;

    const char *pending_env = getenv("MAX_PENDING_SCAN");
    if (pending_env && pending_env[0] != '\0') {
        int requested = atoi(pending_env);
        if (requested >= 1) max_pending_scan = requested;
    }

    SafeQueue<ScanTask> scan_queue((size_t)max_pending_scan);

    // Backpressure for Mongo match queue (candidate -> Mongo logger).
    // 0 means unbounded.
    size_t max_pending_match = 0;
    const char *pending_match_env = getenv("MAX_PENDING_MATCH");
    if (pending_match_env && pending_match_env[0] != '\0') {
        long long requested = atoll(pending_match_env);
        if (requested > 0) max_pending_match = (size_t)requested;
    }
    g_match_queue.setCapacity(max_pending_match);

    const char *logger_mongo_uri = getenv("MONGO_URI");
    const char *logger_mongo_db = getenv("MONGO_DB");
    const char *logger_candidate_collection = getenv("MONGO_CANDIDATE_COLLECTION");
    if (!logger_mongo_uri || logger_mongo_uri[0] == '\0') logger_mongo_uri = "mongodb://127.0.0.1:27017";
    if (!logger_mongo_db || logger_mongo_db[0] == '\0') logger_mongo_db = "ecdsa";
    if (!logger_candidate_collection || logger_candidate_collection[0] == '\0') logger_candidate_collection = "matched_candidates";

    printf("Candidate logger target: %s / %s.%s\n", logger_mongo_uri, logger_mongo_db, logger_candidate_collection);
    std::thread mongo_logger(MongoMatchLogger, logger_mongo_uri, logger_mongo_db, logger_candidate_collection);

    std::thread completion_worker([&completion_queue, &scan_queue, &free_indices, h_r, copy_done_events, X_bytes]() {
        while (true) {
            PendingBatch batch = completion_queue.pop();
            if (batch.exit_signal) break;

            checkCuda(cudaEventSynchronize(copy_done_events[batch.stream_idx]), "wait for copy completion");

            std::shared_ptr<uint64_t> result_copy(new uint64_t[(size_t)batch.n * 4], std::default_delete<uint64_t[]>());
            memcpy(result_copy.get(), h_r[batch.stream_idx], X_bytes);

            free_indices.push(batch.stream_idx);

            ScanTask scan_task;
            memcpy(scan_task.k_start, batch.k_start, sizeof(scan_task.k_start));
            scan_task.r_values = std::move(result_copy);
            scan_task.n = batch.n;
            scan_queue.push(std::move(scan_task));
        }
    });

    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; i++) {
        workers.emplace_back([&scan_queue, &mongo_key_cache, bloom_ptr, scanner_hex_len]() {
            HybridScanner scanner(&mongo_key_cache, bloom_ptr, scanner_hex_len);
            while (true) {
                ScanTask task = scan_queue.pop();
                if (task.exit_signal) break;

                scanner.checkMatchBatch(task.k_start, task.r_values.get(), task.n);
            }
        });
    }

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    printf("Starting Producer loop (GPU) with %d Workers...\n", num_workers);
    printf("PendingScan cap: %d batches\n", max_pending_scan);
    if (max_pending_match > 0) {
        printf("PendingMatch cap: %zu items\n", max_pending_match);
    } else {
        printf("PendingMatch cap: unbounded\n");
    }

    std::atomic<long long> total_keys(0);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    std::seed_seq seed_seq{
        rd(), rd(), rd(), rd(),
        (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count()
    };
    std::mt19937_64 rng(seed_seq);

    auto randomize_k_start = [&]() {
        do {
            k_start[0] = rng();
            k_start[1] = rng();
            k_start[2] = rng();
            k_start[3] = rng();
        } while (k_start[0] == 0 && k_start[1] == 0 && k_start[2] == 0 && k_start[3] == 0);
    };

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
        checkCuda(cudaEventRecord(copy_done_events[idx], streams[idx]), "record copy event");

        // 4. 丟進完成佇列，由完成執行緒在 copy 結束後立刻回收 stream
        PendingBatch batch;
        batch.stream_idx = idx;
        memcpy(batch.k_start, k_start, sizeof(batch.k_start));
        batch.n = n;
        completion_queue.push(std::move(batch));

        total_keys += n;
        static auto last_print = start_time;
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed = current_time - start_time;
        std::chrono::duration<double> print_elapsed = current_time - last_print;

        if (print_elapsed.count() > 2.0) {
             static long long last_scanned_hits = 0;
             static long long last_enqueued = 0;
             static long long last_processed = 0;
             static long long last_precise = 0;

             const double interval_s = print_elapsed.count();
            double speed = total_keys / total_elapsed.count();
            long long scanned_hits = g_scan_candidates.load(std::memory_order_relaxed);
            long long enqueued = g_match_enqueued.load(std::memory_order_relaxed);
            long long processed = g_match_processed.load(std::memory_order_relaxed);
            long long precise = g_match_precise.load(std::memory_order_relaxed);
            long long mongo_us = g_mongo_total_us.load(std::memory_order_relaxed);
            double mongo_avg_ms = (processed > 0) ? ((double)mongo_us / (double)processed / 1000.0) : 0.0;

             double cand_per_sec = (interval_s > 0.0) ? (double)(scanned_hits - last_scanned_hits) / interval_s : 0.0;
             double enq_per_sec = (interval_s > 0.0) ? (double)(enqueued - last_enqueued) / interval_s : 0.0;
             double proc_per_sec = (interval_s > 0.0) ? (double)(processed - last_processed) / interval_s : 0.0;
             double prec_per_sec = (interval_s > 0.0) ? (double)(precise - last_precise) / interval_s : 0.0;

             last_scanned_hits = scanned_hits;
             last_enqueued = enqueued;
             last_processed = processed;
             last_precise = precise;

                 printf("\r[PRODUCER] Speed: %.2f keys/sec | Total: %lld | PendingCopy: %zu | PendingScan: %zu | PendingMatch: %zu | Cand:%lld (%.1f/s) Enq:%lld (%.1f/s) Proc:%lld (%.1f/s) Prec:%lld (%.1f/s) MongoAvg:%.2fms",
                   speed,
                   (long long)total_keys,
                   completion_queue.size(),
                   scan_queue.size(),
                     g_match_queue.size(),
                   scanned_hits,
                 cand_per_sec,
                   enqueued,
                 enq_per_sec,
                   processed,
                 proc_per_sec,
                   precise,
                 prec_per_sec,
                   mongo_avg_ms);
            fflush(stdout);
            last_print = current_time;
        }

        // 第一批使用原始 k_start；後續每批改用隨機起點
        randomize_k_start();

    } while (infinite);

    // Shutdown copy-completion worker after all producer batches are queued.
    PendingBatch completion_sentinel;
    completion_sentinel.exit_signal = true;
    completion_queue.push(std::move(completion_sentinel));
    completion_worker.join();

    // Shutdown scan workers
    for (int i = 0; i < num_workers; i++) {
        ScanTask sentinel;
        sentinel.exit_signal = true;
        scan_queue.push(std::move(sentinel));
    }
    for (auto &t : workers) t.join();

    // Shutdown mongo logger
    MatchData m_sentinel;
    m_sentinel.exit_signal = true;
    g_match_queue.push(m_sentinel);
    mongo_logger.join();

        printf("\n[FINAL] Cand=%lld Enq=%lld Proc=%lld Prec=%lld PendingMatch=%zu\n",
            g_scan_candidates.load(std::memory_order_relaxed),
            g_match_enqueued.load(std::memory_order_relaxed),
            g_match_processed.load(std::memory_order_relaxed),
            g_match_precise.load(std::memory_order_relaxed),
            g_match_queue.size());
        fflush(stdout);

    // Cleanup MongoDB Driver (Global)
    mongoc_cleanup();

    for (int i = 0; i < num_streams; i++) {
        if (h_r[i]) cudaFreeHost(h_r[i]);
        if (d_X[i]) cudaFree(d_X[i]);
        if (d_Y[i]) cudaFree(d_Y[i]);
        if (d_Z[i]) cudaFree(d_Z[i]);
        if (d_r[i]) cudaFree(d_r[i]);
        if (copy_done_events[i]) cudaEventDestroy(copy_done_events[i]);
        if (streams[i]) cudaStreamDestroy(streams[i]);
    }

    return 0;
}
