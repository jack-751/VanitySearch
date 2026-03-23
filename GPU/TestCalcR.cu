/*
 * TestCalcR.cu
 *
 * 單一 k 值測試：計算 r = x(k*G) mod n (secp256k1)
 * 不需要 MongoDB 或 cache 檔。
 *
 * 編譯：
 *   nvcc -O2 -arch=sm_86 -I. TestCalcR.cu -o TestCalcR
 *
 * 執行：
 *   ./TestCalcR <k_decimal_or_hex>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define NBBLOCK  5
#define IDX      threadIdx.x
#define GRP_SIZE 1024
#include "GPUMath.h"

// secp256k1 constants (host)
static const uint64_t HOST_P[4]  = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
static const uint64_t HOST_N[4]  = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
static const uint64_t HOST_GX[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
static const uint64_t HOST_GY[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };

__device__ __constant__ uint64_t DEV_P[4];
__device__ __constant__ uint64_t DEV_N[4];
__device__ __constant__ uint64_t DEV_GX[4];
__device__ __constant__ uint64_t DEV_GY[4];

// --------------------------------------------------------------------------
__device__ __forceinline__ void FullReduceP(uint64_t r[4]) {
    uint64_t d[4] = {0xFFFFFFFEFFFFFC2FULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
    int ge = (r[3]>d[3])||(r[3]==d[3]&&r[2]>d[2])||(r[3]==d[3]&&r[2]==d[2]&&r[1]>d[1])||(r[3]==d[3]&&r[2]==d[2]&&r[1]==d[1]&&r[0]>=d[0]);
    if (ge) ModSub256(r, r, d);
}

__device__ __forceinline__ void ModAddP(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t b_neg[4], bc[4]={b[0],b[1],b[2],b[3]};
    ModNeg256(b_neg, bc);
    uint64_t ac[4]={a[0],a[1],a[2],a[3]};
    ModSub256(r, ac, b_neg);
}
__device__ __forceinline__ void ModDoubleP(uint64_t r[4], const uint64_t a[4]) { ModAddP(r,a,a); }

__device__ __forceinline__ void ModMulP(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t ta[5]={a[0],a[1],a[2],a[3],0}, tb[5]={b[0],b[1],b[2],b[3],0}, res[5];
    _ModMult(res, ta, tb);
    r[0]=res[0]; r[1]=res[1]; r[2]=res[2]; r[3]=res[3];
    FullReduceP(r);
}
__device__ __forceinline__ void ModSqP(uint64_t r[4], const uint64_t a[4]) {
    uint64_t ta[5]={a[0],a[1],a[2],a[3],0}, res[4];
    _ModSqr(res, ta);
    r[0]=res[0]; r[1]=res[1]; r[2]=res[2]; r[3]=res[3];
    FullReduceP(r);
}

// --------------------------------------------------------------------------
__device__ void JacDouble(uint64_t X3[4],uint64_t Y3[4],uint64_t Z3[4],
                          const uint64_t X1[4],const uint64_t Y1[4],const uint64_t Z1[4]) {
    uint64_t Y1sq[4],S[4],M[4],M2[4],Y1_4[4],tmp[4];
    ModSqP(Y1sq,Y1);
    ModMulP(S,X1,Y1sq); ModDoubleP(S,S); ModDoubleP(S,S);
    ModSqP(M,X1); ModDoubleP(tmp,M); ModAddP(M,M,tmp);
    ModSqP(M2,M); ModDoubleP(tmp,S); ModSub256(X3,M2,tmp);
    ModSqP(Y1_4,Y1sq);
    uint64_t ts[4]={S[0],S[1],S[2],S[3]}; ModSub256(tmp,ts,X3); ModMulP(Y3,M,tmp);
    ModDoubleP(Y1_4,Y1_4); ModDoubleP(Y1_4,Y1_4); ModDoubleP(Y1_4,Y1_4);
    uint64_t ty[4]={Y3[0],Y3[1],Y3[2],Y3[3]}; ModSub256(Y3,ty,Y1_4);
    ModMulP(Z3,Y1,Z1); ModDoubleP(Z3,Z3);
}

__device__ void JacMixedAdd(uint64_t X3[4],uint64_t Y3[4],uint64_t Z3[4],
                             const uint64_t X1[4],const uint64_t Y1[4],const uint64_t Z1[4],
                             const uint64_t x2[4],const uint64_t y2[4]) {
    uint64_t Z1sq[4],H[4],R[4],H2[4],H3[4],tmp[4],tmp2[4];
    ModSqP(Z1sq,Z1);
    uint64_t x2c[4]={x2[0],x2[1],x2[2],x2[3]}, y2c[4]={y2[0],y2[1],y2[2],y2[3]};
    uint64_t X1c[4]={X1[0],X1[1],X1[2],X1[3]}, Y1c[4]={Y1[0],Y1[1],Y1[2],Y1[3]};
    ModMulP(H,x2c,Z1sq); { uint64_t ha[4]={H[0],H[1],H[2],H[3]}; ModSub256(H,ha,X1c); }
    uint64_t Z1c2[4]={Z1[0],Z1[1],Z1[2],Z1[3]}, Z1cu[4];
    ModMulP(Z1cu,Z1sq,Z1c2); ModMulP(R,y2c,Z1cu);
    { uint64_t Ra[4]={R[0],R[1],R[2],R[3]}; ModSub256(R,Ra,Y1c); }
    ModSqP(H2,H); ModMulP(H3,H,H2);
    { uint64_t Rsq[4]; ModSqP(Rsq,R);
      { uint64_t Rsqa[4]={Rsq[0],Rsq[1],Rsq[2],Rsq[3]}; ModSub256(tmp,Rsqa,H3); } }
    ModMulP(tmp2,X1c,H2); ModDoubleP(tmp2,tmp2);
    { uint64_t ta[4]={tmp[0],tmp[1],tmp[2],tmp[3]}, tb[4]={tmp2[0],tmp2[1],tmp2[2],tmp2[3]}; ModSub256(X3,ta,tb); }
    ModMulP(tmp,X1c,H2);
    { uint64_t ta[4]={tmp[0],tmp[1],tmp[2],tmp[3]}, X3a[4]={X3[0],X3[1],X3[2],X3[3]}; ModSub256(tmp,ta,X3a); }
    ModMulP(Y3,R,tmp); ModMulP(tmp,Y1c,H3);
    { uint64_t Y3a[4]={Y3[0],Y3[1],Y3[2],Y3[3]}, ta[4]={tmp[0],tmp[1],tmp[2],tmp[3]}; ModSub256(Y3,Y3a,ta); }
    ModMulP(Z3,H,Z1c2);
}

__device__ void ScalarMultG_Jacobian(uint64_t Rx[4],uint64_t Ry[4],uint64_t Rz[4], const uint64_t k[4]) {
    Rx[0]=0;Rx[1]=0;Rx[2]=0;Rx[3]=0;
    Ry[0]=1;Ry[1]=0;Ry[2]=0;Ry[3]=0;
    Rz[0]=0;Rz[1]=0;Rz[2]=0;Rz[3]=0;
    uint64_t Tx[4],Ty[4],Tz[4];
    bool started=false;
    for(int word=3;word>=0;word--) {
        for(int bit=63;bit>=0;bit--) {
            if(started) {
                JacDouble(Tx,Ty,Tz,Rx,Ry,Rz);
                Rx[0]=Tx[0];Rx[1]=Tx[1];Rx[2]=Tx[2];Rx[3]=Tx[3];
                Ry[0]=Ty[0];Ry[1]=Ty[1];Ry[2]=Ty[2];Ry[3]=Ty[3];
                Rz[0]=Tz[0];Rz[1]=Tz[1];Rz[2]=Tz[2];Rz[3]=Tz[3];
            }
            if((k[word]>>bit)&1ULL) {
                if(!started) {
                    Rx[0]=DEV_GX[0];Rx[1]=DEV_GX[1];Rx[2]=DEV_GX[2];Rx[3]=DEV_GX[3];
                    Ry[0]=DEV_GY[0];Ry[1]=DEV_GY[1];Ry[2]=DEV_GY[2];Ry[3]=DEV_GY[3];
                    Rz[0]=1;Rz[1]=0;Rz[2]=0;Rz[3]=0;
                    started=true;
                } else {
                    JacMixedAdd(Tx,Ty,Tz,Rx,Ry,Rz,DEV_GX,DEV_GY);
                    Rx[0]=Tx[0];Rx[1]=Tx[1];Rx[2]=Tx[2];Rx[3]=Tx[3];
                    Ry[0]=Ty[0];Ry[1]=Ty[1];Ry[2]=Ty[2];Ry[3]=Ty[3];
                    Rz[0]=Tz[0];Rz[1]=Tz[1];Rz[2]=Tz[2];Rz[3]=Tz[3];
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Single-thread kernel: compute r = x(k*G) mod n for one k
// --------------------------------------------------------------------------
__global__ void TestSingleK(
    uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3,
    uint64_t *out_r)   // [4]
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t k[4] = {k0, k1, k2, k3};
    uint64_t Rx[4], Ry[4], Rz[4];
    ScalarMultG_Jacobian(Rx, Ry, Rz, k);

    // Affine x = Rx * Rz^{-2} mod P
    uint64_t invZ[5] = {Rz[0], Rz[1], Rz[2], Rz[3], 0};
    _ModInv(invZ);

    uint64_t invZ2[5];
    _ModMult(invZ2, invZ, invZ);

    uint64_t X5[5] = {Rx[0], Rx[1], Rx[2], Rx[3], 0};
    uint64_t Xa5[5];
    _ModMult(Xa5, X5, invZ2);

    // r = x mod n
    uint64_t xa[4] = {Xa5[0], Xa5[1], Xa5[2], Xa5[3]};
    uint64_t local_N[4] = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
    int ge = (xa[3]>local_N[3])||(xa[3]==local_N[3]&&xa[2]>local_N[2])||
             (xa[3]==local_N[3]&&xa[2]==local_N[2]&&xa[1]>local_N[1])||
             (xa[3]==local_N[3]&&xa[2]==local_N[2]&&xa[1]==local_N[1]&&xa[0]>=local_N[0]);
    if (ge) ModSub256(xa, xa, local_N);

    out_r[0]=xa[0]; out_r[1]=xa[1]; out_r[2]=xa[2]; out_r[3]=xa[3];
}

// --------------------------------------------------------------------------
static void parse256(uint64_t r[4], const char *s) {
    memset(r, 0, 32);
    if (!s) return;
    if (s[0]=='0' && (s[1]=='x'||s[1]=='X')) {
        s += 2;
        int len = (int)strlen(s);
        for (int i=0; i<len; i++) {
            int v=0;
            if (s[i]>='0'&&s[i]<='9') v=s[i]-'0';
            else if (s[i]>='a'&&s[i]<='f') v=s[i]-'a'+10;
            else if (s[i]>='A'&&s[i]<='F') v=s[i]-'A'+10;
            else break;
            uint64_t carry=v;
            for (int j=0; j<4; j++) {
                uint64_t nc=r[j]>>60;
                r[j]=(r[j]<<4)+carry;
                carry=nc;
            }
        }
    } else {
        for (int i=0; s[i]!='\0'; i++) {
            if (s[i]<'0'||s[i]>'9') break;
            int digit=s[i]-'0';
            uint64_t carry=digit;
            for (int j=0; j<4; j++) {
                __uint128_t prod=(__uint128_t)r[j]*10+carry;
                r[j]=(uint64_t)prod;
                carry=(uint64_t)(prod>>64);
            }
        }
    }
}

static void printHex256(const char *label, const uint64_t v[4]) {
    printf("%s = %016llx%016llx%016llx%016llx\n", label,
           (unsigned long long)v[3],(unsigned long long)v[2],
           (unsigned long long)v[1],(unsigned long long)v[0]);
}

int main(int argc, char **argv) {
    const char *k_str = (argc > 1) ? argv[1]
        : "22860751503568827944108675187057424959908371263446804931816731781078483855304";

    uint64_t k[4];
    parse256(k, k_str);

    printf("=== TestCalcR: single k → r (secp256k1) ===\n");
    printHex256("k (hex)", k);

    cudaMemcpyToSymbol(DEV_P,  HOST_P,  32);
    cudaMemcpyToSymbol(DEV_N,  HOST_N,  32);
    cudaMemcpyToSymbol(DEV_GX, HOST_GX, 32);
    cudaMemcpyToSymbol(DEV_GY, HOST_GY, 32);

    uint64_t *d_r;
    cudaMalloc(&d_r, 4 * sizeof(uint64_t));

    TestSingleK<<<1, 1>>>(k[0], k[1], k[2], k[3], d_r);
    cudaDeviceSynchronize();

    uint64_t h_r[4];
    cudaMemcpy(h_r, d_r, 32, cudaMemcpyDeviceToHost);
    cudaFree(d_r);

    printHex256("r (hex)", h_r);

    const uint64_t expected[4] = {
        0x0542e03b593d00ddULL, // NOTE: printed big-endian below; stored little-endian
        0xeec1dae4ULL,         // placeholder — recomputed from the full value
        0x0ULL, 0x0ULL
    };
    // Expected: c0199ab7191cd18ccea7e4e9a4fd8ee5eec1dae40542e03b593d00dd1bc81690
    // little-endian limbs:
    //   limb[0] (bits 0-63)  = 0x0542e03b593d00dd  <- wait, let me reparse
    // Full: c0199ab7191cd18c cea7e4e9a4fd8ee5 eec1dae40542e03b 593d00dd1bc81690
    // Actually in little-endian 64-bit:
    //   v[0] = 0x593d00dd1bc81690
    //   v[1] = 0xeec1dae40542e03b
    //   v[2] = 0xcea7e4e9a4fd8ee5
    //   v[3] = 0xc0199ab7191cd18c
    const char *exp_str = "c0199ab7191cd18ccea7e4e9a4fd8ee5eec1dae40542e03b593d00dd1bc81690";
    uint64_t exp[4];
    {
        char tmp[67] = "0x";
        strncpy(tmp+2, exp_str, 64); tmp[66]='\0';
        parse256(exp, tmp);
    }

    printf("expected r = %016llx%016llx%016llx%016llx\n",
           (unsigned long long)exp[3],(unsigned long long)exp[2],
           (unsigned long long)exp[1],(unsigned long long)exp[0]);

    int match = (h_r[0]==exp[0] && h_r[1]==exp[1] && h_r[2]==exp[2] && h_r[3]==exp[3]);
    printf("\n結果: %s\n", match ? "✓ MATCH — r 值正確！" : "✗ MISMATCH — r 值不符！");
    return match ? 0 : 1;
}
