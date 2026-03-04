/*
 * CalcR_kG.cu
 *
 * 在 ECDSA (secp256k1) 中，簽名 r 值的定義為：
 *   r = x_coord( k * G ) mod n
 *
 * 本程式利用 VanitySearch 已有的 GPU 256-bit 模數運算核心
 * (GPUMath.h 的 _ModMult / _ModSqr / _ModInv)，
 * 以 secp256k1 仿射座標直接計算 k=1,2,3 時的 r 值。
 *
 * 編譯方式 (在 VanitySearch\GPU\ 目錄下執行):
 *   nvcc -O2 -arch=sm_86 -I. CalcR_kG.cu -o CalcR_kG.exe
 *   (sm_86 = RTX 3xxx / Ada; 請依卡型調整)
 *
 * 執行後會在 console 列印 k=1,2,3 的 r 值 (十六進位)。
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>


// ===========================================================================
// secp256k1 曲線參數 (host side, 小端排列 64-bit limbs)
// ===========================================================================

// P = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// 以 [limb0(LSB) ... limb3(MSB)] 表示
static const uint64_t P[4] = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                              0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};

// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
static const uint64_t N[4] = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
                              0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};

// G.x = 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
static const uint64_t Gx[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
                               0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};

// G.y = 483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
static const uint64_t Gy[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
                               0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};

// ===========================================================================
// Host-side 256-bit 模數運算 (簡易版，用於驗證)
// ===========================================================================

// 128-bit carry-safe add; ret carry in *c
static inline uint64_t add64(uint64_t a, uint64_t b, uint64_t *carry) {
  __uint128_t s = (__uint128_t)a + b + *carry;
  *carry = (uint64_t)(s >> 64);
  return (uint64_t)s;
}
static inline uint64_t sub64(uint64_t a, uint64_t b, uint64_t *borrow) {
  __uint128_t d = (__uint128_t)a - b - *borrow;
  *borrow = (uint64_t)(-(int64_t)(d >> 64));
  return (uint64_t)d;
}

// a -= b (mod P); a, b are 4x uint64 little-endian
static void modSubP(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t borrow = 0;
  r[0] = sub64(a[0], b[0], &borrow);
  r[1] = sub64(a[1], b[1], &borrow);
  r[2] = sub64(a[2], b[2], &borrow);
  r[3] = sub64(a[3], b[3], &borrow);
  // if borrow, add P
  if (borrow) {
    uint64_t carry = 0;
    r[0] = add64(r[0], P[0], &carry);
    r[1] = add64(r[1], P[1], &carry);
    r[2] = add64(r[2], P[2], &carry);
    r[3] = add64(r[3], P[3], &carry);
  }
}

// 512-bit * reduction for secp256k1 P
// secp256k1: P = 2^256 - 2^32 - 977, i.e. P ≡ -4294968273 (mod 2^256)
// so  high * 2^256 ≡ high * (2^32 + 977) (mod P)
static void modReduceP(uint64_t r[4], const uint64_t hi[4],
                       const uint64_t lo[4]) {
  // Compute t = hi * (2^32 + 977) as 320-bit
  // 0x1000003D1 = 2^32 + 977
  const uint64_t COEFF = 0x1000003D1ULL;
  __uint128_t acc;
  uint64_t t[5];
  acc = (__uint128_t)hi[0] * COEFF;
  t[0] = (uint64_t)acc;
  acc >>= 64;
  acc += (__uint128_t)hi[1] * COEFF;
  t[1] = (uint64_t)acc;
  acc >>= 64;
  acc += (__uint128_t)hi[2] * COEFF;
  t[2] = (uint64_t)acc;
  acc >>= 64;
  acc += (__uint128_t)hi[3] * COEFF;
  t[3] = (uint64_t)acc;
  acc >>= 64;
  t[4] = (uint64_t)acc;

  // r = lo + t (mod P), with potential carry from t[4]
  uint64_t carry = 0;
  r[0] = add64(lo[0], t[0], &carry);
  r[1] = add64(lo[1], t[1], &carry);
  r[2] = add64(lo[2], t[2], &carry);
  r[3] = add64(lo[3], t[3], &carry);
  carry += t[4];

  // handle the small overflow (carry * 2^256)
  if (carry) {
    __uint128_t extra = (__uint128_t)carry * COEFF;
    uint64_t e0 = (uint64_t)extra;
    uint64_t e1 = (uint64_t)(extra >> 64);
    uint64_t c2 = 0;
    r[0] = add64(r[0], e0, &c2);
    r[1] = add64(r[1], e1, &c2);
    r[2] = add64(r[2], 0, &c2);
    r[3] = add64(r[3], 0, &c2);
  }

  // subtract P once if r >= P
  // check r >= P
  int ge = (r[3] > P[3]) || (r[3] == P[3] && r[2] > P[2]) ||
           (r[3] == P[3] && r[2] == P[2] && r[1] > P[1]) ||
           (r[3] == P[3] && r[2] == P[2] && r[1] == P[1] && r[0] >= P[0]);
  if (ge) {
    uint64_t borrow = 0;
    r[0] = sub64(r[0], P[0], &borrow);
    r[1] = sub64(r[1], P[1], &borrow);
    r[2] = sub64(r[2], P[2], &borrow);
    r[3] = sub64(r[3], P[3], &borrow);
  }
}

// r = a * b mod P  (256x256 -> 512 -> reduce)
static void modMultP(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
  uint64_t lo[4], hi[4];
  // schoolbook 256x256 = 512-bit product
  __uint128_t cols[8] = {};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      cols[i + j] += (__uint128_t)a[i] * b[j];

  uint64_t p[8];
  __uint128_t carry = 0;
  for (int k = 0; k < 8; k++) {
    cols[k] += carry;
    p[k] = (uint64_t)cols[k];
    carry = cols[k] >> 64;
  }
  memcpy(lo, p, 32);
  memcpy(hi, p + 4, 32);
  modReduceP(r, hi, lo);
}

// r = a^2 mod P
static void modSqrP(uint64_t r[4], const uint64_t a[4]) { modMultP(r, a, a); }

// Modular inverse: r = a^{-1} mod P  (Fermat: a^{P-2} mod P)
// P-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
static void modInvP(uint64_t r[4], const uint64_t a[4]) {
  // Uses square-and-multiply with the exponent P-2.
  // We follow the efficient chain for secp256k1 from standard references.
  // Binary exponentiation on P-2:
  uint64_t exp[4] = {0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL,
                     0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};

  uint64_t base[4], res[4] = {1, 0, 0, 0};
  memcpy(base, a, 32);

  for (int word = 0; word < 4; word++) {
    for (int bit = 0; bit < 64; bit++) {
      if ((exp[word] >> bit) & 1ULL) {
        modMultP(res, res, base);
      }
      modSqrP(base, base);
    }
  }
  memcpy(r, res, 32);
}

// ===========================================================================
// secp256k1 Affine Point Addition  P3 = P1 + P2  (both ≠ infinity)
// ===========================================================================
static void pointAdd(uint64_t rx[4], uint64_t ry[4], const uint64_t p1x[4],
                     const uint64_t p1y[4], const uint64_t p2x[4],
                     const uint64_t p2y[4]) {
  // s = (p2y - p1y) / (p2x - p1x) mod P
  uint64_t dy[4], dx[4], dxInv[4], s[4], s2[4], tmp[4];

  modSubP(dy, p2y, p1y);
  modSubP(dx, p2x, p1x);
  modInvP(dxInv, dx);
  modMultP(s, dy, dxInv);

  // rx = s^2 - p1x - p2x
  modSqrP(s2, s);
  modSubP(tmp, s2, p1x);
  modSubP(rx, tmp, p2x);

  // ry = s*(p1x - rx) - p1y
  modSubP(tmp, p1x, rx);
  modMultP(ry, s, tmp);
  modSubP(ry, ry, p1y);
}

// secp256k1 Affine Point Doubling  P3 = 2*P1
static void pointDouble(uint64_t rx[4], uint64_t ry[4], const uint64_t px[4],
                        const uint64_t py[4]) {
  // s = 3*px^2 / (2*py)  (a=0)
  uint64_t px2[4], s[4], s2[4], tmp[4], twoInv[4];

  modSqrP(px2, px); // px^2
  // 3*px^2 mod P
  uint64_t threeX2[4];
  {
    uint64_t carry = 0;
    threeX2[0] = add64(px2[0], px2[0], &carry);
    threeX2[1] = add64(px2[1], px2[1], &carry);
    threeX2[2] = add64(px2[2], px2[2], &carry);
    threeX2[3] = add64(px2[3], px2[3], &carry);
    // reduce if >= P
    int ge = carry || (threeX2[3] > P[3]) ||
             (threeX2[3] == P[3] && threeX2[2] > P[2]) ||
             (threeX2[3] == P[3] && threeX2[2] == P[2] && threeX2[1] > P[1]) ||
             (threeX2[3] == P[3] && threeX2[2] == P[2] && threeX2[1] == P[1] &&
              threeX2[0] >= P[0]);
    if (ge) {
      uint64_t borrow = 0;
      threeX2[0] = sub64(threeX2[0], P[0], &borrow);
      threeX2[1] = sub64(threeX2[1], P[1], &borrow);
      threeX2[2] = sub64(threeX2[2], P[2], &borrow);
      threeX2[3] = sub64(threeX2[3], P[3], &borrow);
    }
    carry = 0;
    uint64_t t3[4];
    t3[0] = add64(threeX2[0], px2[0], &carry);
    t3[1] = add64(threeX2[1], px2[1], &carry);
    t3[2] = add64(threeX2[2], px2[2], &carry);
    t3[3] = add64(threeX2[3], px2[3], &carry);
    ge = carry || (t3[3] > P[3]) || (t3[3] == P[3] && t3[2] > P[2]) ||
         (t3[3] == P[3] && t3[2] == P[2] && t3[1] > P[1]) ||
         (t3[3] == P[3] && t3[2] == P[2] && t3[1] == P[1] && t3[0] >= P[0]);
    if (ge) {
      uint64_t borrow = 0;
      t3[0] = sub64(t3[0], P[0], &borrow);
      t3[1] = sub64(t3[1], P[1], &borrow);
      t3[2] = sub64(t3[2], P[2], &borrow);
      t3[3] = sub64(t3[3], P[3], &borrow);
    }
    memcpy(threeX2, t3, 32);
  }

  // 2*py
  uint64_t twoPy[4];
  {
    uint64_t carry = 0;
    twoPy[0] = add64(py[0], py[0], &carry);
    twoPy[1] = add64(py[1], py[1], &carry);
    twoPy[2] = add64(py[2], py[2], &carry);
    twoPy[3] = add64(py[3], py[3], &carry);
    int ge = carry || (twoPy[3] > P[3]) ||
             (twoPy[3] == P[3] && twoPy[2] > P[2]) ||
             (twoPy[3] == P[3] && twoPy[2] == P[2] && twoPy[1] > P[1]) ||
             (twoPy[3] == P[3] && twoPy[2] == P[2] && twoPy[1] == P[1] &&
              twoPy[0] >= P[0]);
    if (ge) {
      uint64_t borrow = 0;
      twoPy[0] = sub64(twoPy[0], P[0], &borrow);
      twoPy[1] = sub64(twoPy[1], P[1], &borrow);
      twoPy[2] = sub64(twoPy[2], P[2], &borrow);
      twoPy[3] = sub64(twoPy[3], P[3], &borrow);
    }
  }
  modInvP(twoInv, twoPy);
  modMultP(s, threeX2, twoInv);

  // rx = s^2 - 2*px
  modSqrP(s2, s);
  modSubP(tmp, s2, px);
  modSubP(rx, tmp, px);

  // ry = s*(px - rx) - py
  modSubP(tmp, px, rx);
  modMultP(ry, s, tmp);
  modSubP(ry, ry, py);
}

// r = x mod n  (x is already reduced mod P < n for secp256k1 is not guaranteed,
// but since P < n, we just check and subtract n if needed)
static void modN(uint64_t r[4], const uint64_t x[4]) {
  // check x >= N
  int ge = (x[3] > N[3]) || (x[3] == N[3] && x[2] > N[2]) ||
           (x[3] == N[3] && x[2] == N[2] && x[1] > N[1]) ||
           (x[3] == N[3] && x[2] == N[2] && x[1] == N[1] && x[0] >= N[0]);
  if (ge) {
    uint64_t borrow = 0;
    r[0] = sub64(x[0], N[0], &borrow);
    r[1] = sub64(x[1], N[1], &borrow);
    r[2] = sub64(x[2], N[2], &borrow);
    r[3] = sub64(x[3], N[3], &borrow);
  } else {
    memcpy(r, x, 32);
  }
}

static void printHex256(const char *label, const uint64_t v[4]) {
  // Print as big-endian hex (most-significant first)
  printf("%s = %016llx%016llx%016llx%016llx\n", label, (unsigned long long)v[3],
         (unsigned long long)v[2], (unsigned long long)v[1],
         (unsigned long long)v[0]);
}

int main() {
  printf("=== ECDSA secp256k1  r = x( k*G ) mod n ===\n\n");

  printf("secp256k1 Generator G:\n");
  printHex256("  Gx", Gx);
  printHex256("  Gy", Gy);
  printf("\n");

  /*
   * k=1 : 1*G = G itself, so r = Gx mod n
   * k=2 : 2*G = double(G)
   * k=3 : 3*G = 2*G + G
   */

  uint64_t rx[4], ry[4], tmp_x[4], tmp_y[4], r_val[4];

  // -------------------------------------------------------
  // k = 1
  // -------------------------------------------------------
  modN(r_val, Gx); // Gx is already < n for secp256k1
  printf("k = 1:\n");
  printHex256("  k*G  .x", Gx);
  printHex256("  r = x(kG) mod n", r_val);
  printf("\n");

  // -------------------------------------------------------
  // k = 2:  2*G = doublePoint(G)
  // -------------------------------------------------------
  pointDouble(rx, ry, Gx, Gy);
  modN(r_val, rx);
  printf("k = 2:\n");
  printHex256("  k*G  .x", rx);
  printHex256("  k*G  .y", ry);
  printHex256("  r = x(kG) mod n", r_val);
  printf("\n");

  // -------------------------------------------------------
  // k = 3:  3*G = 2*G + G
  // -------------------------------------------------------
  memcpy(tmp_x, rx, 32);
  memcpy(tmp_y, ry, 32);
  pointAdd(rx, ry, tmp_x, tmp_y, Gx, Gy);
  modN(r_val, rx);
  printf("k = 3:\n");
  printHex256("  k*G  .x", rx);
  printHex256("  k*G  .y", ry);
  printHex256("  r = x(kG) mod n", r_val);
  printf("\n");

  printf("=== Done ===\n");
  return 0;
}
