//
// Created by Marco.LIU on 16/8/19.
// Copyright (c) 2016 LLX. All rights reserved.
//

#ifndef _LLX_UTIL_MATRIX_MATRICES_INLINE_H_
#define _LLX_UTIL_MATRIX_MATRICES_INLINE_H_

#include "matrix_config.h"

#if NO_INLINE
#include "matrices.h"
#include "global_consts.h"
#endif

#include <assert.h>

namespace LLX {
namespace util {
LLX_INLINE Matrix::Matrix(float m00, float m10, float m20, float m30,
                          float m01, float m11, float m21, float m31,
                          float m02, float m12, float m22, float m32,
                          float m03, float m13, float m23, float m33) {
  c[0] = VectorSet(m00, m10, m20, m30);
  c[1] = VectorSet(m01, m11, m21, m31);
  c[2] = VectorSet(m02, m12, m22, m32);
  c[3] = VectorSet(m03, m13, m23, m33);
}

LLX_INLINE Matrix Matrix::operator-() const {
  Matrix m;
  m.c[0] = VectorNegate(c[0]);
  m.c[1] = VectorNegate(c[1]);
  m.c[2] = VectorNegate(c[2]);
  m.c[3] = VectorNegate(c[3]);
  return m;
}

LLX_INLINE Matrix& Matrix::operator+= (FMatrix m) {
  c[0] = VectorAdd(c[0], m.c[0]);
  c[1] = VectorAdd(c[1], m.c[1]);
  c[2] = VectorAdd(c[2], m.c[2]);
  c[3] = VectorAdd(c[3], m.c[3]);
  return *this;
}
LLX_INLINE Matrix& Matrix::operator-= (FMatrix m) {
  c[0] = VectorSubtract(c[0], m.c[0]);
  c[1] = VectorSubtract(c[1], m.c[1]);
  c[2] = VectorSubtract(c[2], m.c[2]);
  c[3] = VectorSubtract(c[3], m.c[3]);
  return *this;
}
LLX_INLINE Matrix& Matrix::operator*= (FMatrix m) {
  *this = MatrixMultiply(*this, m);
  return *this;
}
LLX_INLINE Matrix& Matrix::operator*= (float s) {
  c[0] = VectorScale(c[0], s);
  c[1] = VectorScale(c[1], s);
  c[2] = VectorScale(c[2], s);
  c[3] = VectorScale(c[3], s);
  return *this;
}
LLX_INLINE Matrix& Matrix::operator/= (float s) {
#if USE_ARM_NEON
  float32x2_t vS = vdup_n_f32(s);
  float32x2_t R0 = vrecpe_f32(vS);
  float32x2_t S0 = vrecps_f32(R0, vS);
  R0 = vmul_f32(S0, R0);
  S0 = vrecps_f32(R0, vS);
  R0 = vmul_f32(S0, R0);
  float32x4_t Reciprocal = vcombine_f32(R0, R0);
  c[0] = vmulq_f32(c[0], Reciprocal);
  c[1] = vmulq_f32(c[1], Reciprocal);
  c[2] = vmulq_f32(c[2], Reciprocal);
  c[3] = vmulq_f32(c[3], Reciprocal);
  return *this;
#else
  float rs = 1.f / s;
  return this->operator*=(rs);
#endif
}

LLX_INLINE Matrix Matrix::operator+ (FMatrix m) const {
  Matrix r;
  r.c[0] = VectorAdd(c[0], m.c[0]);
  r.c[1] = VectorAdd(c[1], m.c[1]);
  r.c[2] = VectorAdd(c[2], m.c[2]);
  r.c[3] = VectorAdd(c[3], m.c[3]);
  return r;
}
LLX_INLINE Matrix Matrix::operator- (FMatrix m) const {
  Matrix r;
  r.c[0] = VectorSubtract(c[0], m.c[0]);
  r.c[1] = VectorSubtract(c[1], m.c[1]);
  r.c[2] = VectorSubtract(c[2], m.c[2]);
  r.c[3] = VectorSubtract(c[3], m.c[3]);
  return r;
}
LLX_INLINE Matrix Matrix::operator* (FMatrix m) const {
  return MatrixMultiply(*this, m);
}
LLX_INLINE Matrix Matrix::operator* (float s) const {
  Matrix r;
  r.c[0] = VectorScale(c[0], s);
  r.c[1] = VectorScale(c[1], s);
  r.c[2] = VectorScale(c[2], s);
  r.c[3] = VectorScale(c[3], s);
  return r;
}
LLX_INLINE Matrix Matrix::operator/ (float s) const {
#if USE_ARM_NEON
  float32x2_t vS = vdup_n_f32(s);
  float32x2_t R0 = vrecpe_f32(vS);
  float32x2_t S0 = vrecps_f32(R0, vS);
  R0 = vmul_f32(S0, R0);
  S0 = vrecps_f32(R0, vS);
  R0 = vmul_f32(S0, R0);
  float32x4_t Reciprocal = vcombine_f32(R0, R0);
  Matrix r;
  r.c[0] = vmulq_f32(c[0], Reciprocal);
  r.c[1] = vmulq_f32(c[1], Reciprocal);
  r.c[2] = vmulq_f32(c[2], Reciprocal);
  r.c[3] = vmulq_f32(c[3], Reciprocal);
  return r;
#else
  float rs = 1.f / s;
  return this->operator*(rs);
#endif
}

LLX_INLINE Matrix operator* (float s, FMatrix m) {
  Matrix r;
  r.c[0] = VectorScale(m.c[0], s);
  r.c[1] = VectorScale(m.c[1], s);
  r.c[2] = VectorScale(m.c[2], s);
  r.c[3] = VectorScale(m.c[3], s);
  return r;
}

LLX_INLINE bool MatrixIsNaN(FMatrix m) {
#if USE_ARM_NEON
  // Load in registers
  Vector vX = m.c[0];
  Vector vY = m.c[1];
  Vector vZ = m.c[2];
  Vector vW = m.c[3];
  // Test themselves to check for NaN
  vX = vmvnq_u32(vceqq_f32(vX, vX));
  vY = vmvnq_u32(vceqq_f32(vY, vY));
  vZ = vmvnq_u32(vceqq_f32(vZ, vZ));
  vW = vmvnq_u32(vceqq_f32(vW, vW));
  // Or all the results
  vX = vorrq_u32(vX, vZ);
  vY = vorrq_u32(vY, vW);
  vX = vorrq_u32(vX, vY);
  // If any tested true, return true
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vX), vget_high_u8(vX));
  uint16x4x2_t vTemp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  uint32_t r = vget_lane_u32(vTemp0.val[1], 1);
  return (r != 0);
#else
  size_t i = 16;
  const uint32_t *pWork = (const uint32_t *)(&m.m[0][0]);
  do {
    // Fetch value into integer unit
    uint32_t uTest = pWork[0];
    // Remove sign
    uTest &= 0x7FFFFFFFU;
    // NaN is 0x7F800001 through 0x7FFFFFFF inclusive
    uTest -= 0x7F800001U;
    if (uTest<0x007FFFFFU) {
      break;      // NaN found
    }
    ++pWork;        // Next entry
  } while (--i);
  return (i!=0);      // i == 0 if nothing matched
#endif
}
LLX_INLINE bool MatrixIsInfinite(FMatrix m) {
#if USE_ARM_NEON
  // Mask off the sign bits
  Vector vTemp1 = vandq_u32(m.c[0], kAbsMask);
  Vector vTemp2 = vandq_u32(m.c[1], kAbsMask);
  Vector vTemp3 = vandq_u32(m.c[2], kAbsMask);
  Vector vTemp4 = vandq_u32(m.c[3], kAbsMask);
  // Compare to infinity
  vTemp1 = vceqq_f32(vTemp1, kInfinity);
  vTemp2 = vceqq_f32(vTemp2, kInfinity);
  vTemp3 = vceqq_f32(vTemp3, kInfinity);
  vTemp4 = vceqq_f32(vTemp4, kInfinity);
  // Or the answers together
  vTemp1 = vorrq_u32(vTemp1, vTemp2);
  vTemp3 = vorrq_u32(vTemp3, vTemp4);
  vTemp1 = vorrq_u32(vTemp1, vTemp3);
  // If any are infinity, the signs are true.
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vTemp1), vget_high_u8(vTemp1));
  uint16x4x2_t vTemp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  uint32_t r = vget_lane_u32(vTemp0.val[1], 1);
  return (r != 0);
#else
  size_t i = 16;
  const uint32_t *pWork = (const uint32_t *)(&m.m[0][0]);
  do {
      // Fetch value into integer unit
      uint32_t uTest = pWork[0];
      // Remove sign
      uTest &= 0x7FFFFFFFU;
      // INF is 0x7F800000
      if (uTest==0x7F800000U) {
          break;      // INF found
      }
      ++pWork;        // Next entry
  } while (--i);
  return (i!=0);      // i == 0 if nothing matched
#endif
}
LLX_INLINE bool MatrixIsIdentity(FMatrix m) {
#if USE_ARM_NEON
  Vector vTemp1 = vceqq_f32(m.c[0], kIdentityX);
  Vector vTemp2 = vceqq_f32(m.c[1], kIdentityY);
  Vector vTemp3 = vceqq_f32(m.c[2], kIdentityZ);
  Vector vTemp4 = vceqq_f32(m.c[3], kIdentityW);
  vTemp1 = vandq_u32(vTemp1, vTemp2);
  vTemp3 = vandq_u32(vTemp3, vTemp4);
  vTemp1 = vandq_u32(vTemp1, vTemp3);
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vTemp1), vget_high_u8(vTemp1));
  uint16x4x2_t vTemp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  uint32_t r = vget_lane_u32(vTemp0.val[1], 1);
  return (r == 0xFFFFFFFFU);
#else
  // Use the integer pipeline to reduce branching to a minimum
  const uint32_t *pWork = (const uint32_t*)(&m.m[0][0]);
  // Convert 1.0f to zero and or them together
  uint32_t uOne = pWork[0]^0x3F800000U;
  // Or all the 0.0f entries together
  uint32_t uZero = pWork[1];
  uZero |= pWork[2];
  uZero |= pWork[3];
  // 2nd row
  uZero |= pWork[4];
  uOne |= pWork[5]^0x3F800000U;
  uZero |= pWork[6];
  uZero |= pWork[7];
  // 3rd row
  uZero |= pWork[8];
  uZero |= pWork[9];
  uOne |= pWork[10]^0x3F800000U;
  uZero |= pWork[11];
  // 4th row
  uZero |= pWork[12];
  uZero |= pWork[13];
  uZero |= pWork[14];
  uOne |= pWork[15]^0x3F800000U;
  // If all zero entries are zero, the uZero==0
  uZero &= 0x7FFFFFFF;    // Allow -0.0f
  // If all 1.0f entries are 1.0f, then uOne==0
  uOne |= uZero;
  return (uOne==0);
#endif
}
LLX_INLINE Matrix MatrixMultiply(FMatrix m1, CMatrix m2) {
  Matrix r;
#if USE_ARM_NEON
  float32x2_t VL = vget_low_f32(m2.c[0]);
  float32x2_t VH = vget_high_f32(m2.c[0]);
  // Perform the operation on the first row
  Vector vX = vmulq_lane_f32(m1.c[0], VL, 0);
  Vector vY = vmulq_lane_f32(m1.c[1], VL, 1);
  Vector vZ = vmlaq_lane_f32(vX, m1.c[2], VH, 0);
  Vector vW = vmlaq_lane_f32(vY, m1.c[3], VH, 1);
  r.c[0] = vaddq_f32(vZ, vW);
  // Repeat for the other 3 rows
  VL = vget_low_f32(m2.c[1]);
  VH = vget_high_f32(m2.c[1]);
  vX = vmulq_lane_f32(m1.c[0], VL, 0);
  vY = vmulq_lane_f32(m1.c[1], VL, 1);
  vZ = vmlaq_lane_f32(vX, m1.c[2], VH, 0);
  vW = vmlaq_lane_f32(vY, m1.c[3], VH, 1);
  r.c[1] = vaddq_f32(vZ, vW);
  VL = vget_low_f32(m2.c[2]);
  VH = vget_high_f32(m2.c[2]);
  vX = vmulq_lane_f32(m1.c[0], VL, 0);
  vY = vmulq_lane_f32(m1.c[1], VL, 1);
  vZ = vmlaq_lane_f32(vX, m1.c[2], VH, 0);
  vW = vmlaq_lane_f32(vY, m1.c[3], VH, 1);
  r.c[2] = vaddq_f32(vZ, vW);
  VL = vget_low_f32(m2.c[3]);
  VH = vget_high_f32(m2.c[3]);
  vX = vmulq_lane_f32(m1.c[0], VL, 0);
  vY = vmulq_lane_f32(m1.c[1], VL, 1);
  vZ = vmlaq_lane_f32(vX, m1.c[2], VH, 0);
  vW = vmlaq_lane_f32(vY, m1.c[3], VH, 1);
  r.c[3] = vaddq_f32(vZ, vW);
#else
  float x = m1._00;
  float y = m1._01;
  float z = m1._02;
  float w = m1._03;
  r._00 = x * m2._00 + y * m2._10 + z * m2._20 + w * m2._30;
  r._01 = x * m2._01 + y * m2._11 + z * m2._21 + w * m2._31;
  r._02 = x * m2._02 + y * m2._12 + z * m2._22 + w * m2._32;
  r._03 = x * m2._03 + y * m2._13 + z * m2._23 + w * m2._33;
  
  x = m1._10;
  y = m1._11;
  z = m1._12;
  w = m1._13;
  r._10 = x * m2._00 + y * m2._10 + z * m2._20 + w * m2._30;
  r._11 = x * m2._01 + y * m2._11 + z * m2._21 + w * m2._31;
  r._12 = x * m2._02 + y * m2._12 + z * m2._22 + w * m2._32;
  r._13 = x * m2._03 + y * m2._13 + z * m2._23 + w * m2._33;
  
  x = m1._20;
  y = m1._21;
  z = m1._22;
  w = m1._23;
  r._20 = x * m2._00 + y * m2._10 + z * m2._20 + w * m2._30;
  r._21 = x * m2._01 + y * m2._11 + z * m2._21 + w * m2._31;
  r._22 = x * m2._02 + y * m2._12 + z * m2._22 + w * m2._32;
  r._23 = x * m2._03 + y * m2._13 + z * m2._23 + w * m2._33;
  
  x = m1._30;
  y = m1._31;
  z = m1._32;
  w = m1._33;
  r._30 = x * m2._00 + y * m2._10 + z * m2._20 + w * m2._30;
  r._31 = x * m2._01 + y * m2._11 + z * m2._21 + w * m2._31;
  r._32 = x * m2._02 + y * m2._12 + z * m2._22 + w * m2._32;
  r._33 = x * m2._03 + y * m2._13 + z * m2._23 + w * m2._33;
#endif
  return r;
}
LLX_INLINE Matrix MatrixTranspose(FMatrix m) {
#if USE_ARM_NEON
  // P0 = {(a00, a02, a10, a12), (a20, a22, a30, a32)}
  float32x4x2_t P0 = vzipq_f32(m.c[0], m.c[2]);
  // P1 = {(a01, a03, a11, a13), (a21, a23, a31, a33)}
  float32x4x2_t P1 = vzipq_f32(m.c[1], m.c[3]);

  // T0 = {(a00, a01, a02, a03), (a10, a11, a12, a13)}
  float32x4x2_t T0 = vzipq_f32(P0.val[0], P1.val[0]);
  // T1 = {(a20, a21, a22, a23), (a30, a31, a32, a33)}
  float32x4x2_t T1 = vzipq_f32(P0.val[1], P1.val[1]);

  Matrix r;
  r.c[0] = T0.val[0];
  r.c[1] = T0.val[1];
  r.c[2] = T1.val[0];
  r.c[3] = T1.val[1];
  return r;
#else
  Matrix r;
  for (int i = 0; i < 4; ++i) {
    for(int j = 0; j < 4; ++j) {
      r.m[i][j] = m.m[j][i];
    }
  }
  return r;
#endif
}
LLX_INLINE Matrix MatrixMultiplyTranspose(FMatrix m1, FMatrix m2) {
  return MatrixTranspose(MatrixMultiply(m1, m2));
}
LLX_INLINE Matrix MatrixInverse(Vector* determinant, FMatrix m) {
  Matrix mt = MatrixTranspose(m);
  Vector V0[4], V1[4];
  V0[0] = VectorSwizzle<kSwizzleX, kSwizzleX, kSwizzleY, kSwizzleY>(mt.c[2]);
  V1[0] = VectorSwizzle<kSwizzleZ, kSwizzleW, kSwizzleZ, kSwizzleW>(mt.c[3]);
  V0[1] = VectorSwizzle<kSwizzleX, kSwizzleX, kSwizzleY, kSwizzleY>(mt.c[0]);
  V1[1] = VectorSwizzle<kSwizzleZ, kSwizzleW, kSwizzleZ, kSwizzleW>(mt.c[1]);
  V0[2] = VectorPermute<kPermute0X, kPermute0Z,
                        kPermute1X, kPermute1Z>(mt.c[2], mt.c[0]);
  V1[2] = VectorPermute<kPermute0Y, kPermute0W,
                        kPermute1Y, kPermute1W>(mt.c[3], mt.c[1]);

  Vector D0 = VectorMultiply(V0[0], V1[0]);
  Vector D1 = VectorMultiply(V0[1], V1[1]);
  Vector D2 = VectorMultiply(V0[2], V1[2]);

  V0[0] = VectorSwizzle<kSwizzleZ, kSwizzleW, kSwizzleZ, kSwizzleW>(mt.c[2]);
  V1[0] = VectorSwizzle<kSwizzleX, kSwizzleX, kSwizzleY, kSwizzleY>(mt.c[3]);
  V0[1] = VectorSwizzle<kSwizzleZ, kSwizzleW, kSwizzleZ, kSwizzleW>(mt.c[0]);
  V1[1] = VectorSwizzle<kSwizzleX, kSwizzleX, kSwizzleY, kSwizzleY>(mt.c[1]);
  V0[2] = VectorPermute<kPermute0Y, kPermute0W,
                        kPermute1Y, kPermute1W>(mt.c[2], mt.c[0]);
  V1[2] = VectorPermute<kPermute0X, kPermute0Z,
                        kPermute1X, kPermute1Z>(mt.c[3], mt.c[1]);

  D0 = VectorNegativeMultiplySubtract(V0[0], V1[0], D0);
  D1 = VectorNegativeMultiplySubtract(V0[1], V1[1], D1);
  D2 = VectorNegativeMultiplySubtract(V0[2], V1[2], D2);

  V0[0] = VectorSwizzle<kSwizzleY, kSwizzleZ, kSwizzleX, kSwizzleY>(mt.c[1]);
  V1[0] = VectorPermute<kPermute1Y, kPermute0Y, kPermute0W, kPermute0X>(D0, D2);
  V0[1] = VectorSwizzle<kSwizzleZ, kSwizzleX, kSwizzleY, kSwizzleX>(mt.c[0]);
  V1[1] = VectorPermute<kPermute0W, kPermute1Y, kPermute0Y, kPermute0Z>(D0, D2);
  V0[2] = VectorSwizzle<kSwizzleY, kSwizzleZ, kSwizzleX, kSwizzleY>(mt.c[3]);
  V1[2] = VectorPermute<kPermute1W, kPermute0Y, kPermute0W, kPermute0X>(D1, D2);
  V0[3] = VectorSwizzle<kSwizzleZ, kSwizzleX, kSwizzleY, kSwizzleX>(mt.c[2]);
  V1[3] = VectorPermute<kPermute0W, kPermute1W, kPermute0Y, kPermute0Z>(D1, D2);

  Vector C0 = VectorMultiply(V0[0], V1[0]);
  Vector C2 = VectorMultiply(V0[1], V1[1]);
  Vector C4 = VectorMultiply(V0[2], V1[2]);
  Vector C6 = VectorMultiply(V0[3], V1[3]);

  V0[0] = VectorSwizzle<kSwizzleZ, kSwizzleW, kSwizzleY, kSwizzleZ>(mt.c[1]);
  V1[0] = VectorPermute<kPermute0W, kPermute0X, kPermute0Y, kPermute1X>(D0, D2);
  V0[1] = VectorSwizzle<kSwizzleW, kSwizzleZ, kSwizzleW, kSwizzleY>(mt.c[0]);
  V1[1] = VectorPermute<kPermute0Z, kPermute0Y, kPermute1X, kPermute0X>(D0, D2);
  V0[2] = VectorSwizzle<kSwizzleZ, kSwizzleW, kSwizzleY, kSwizzleZ>(mt.c[3]);
  V1[2] = VectorPermute<kPermute0W, kPermute0X, kPermute0Y, kPermute1Z>(D1, D2);
  V0[3] = VectorSwizzle<kSwizzleW, kSwizzleZ, kSwizzleW, kSwizzleY>(mt.c[2]);
  V1[3] = VectorPermute<kPermute0Z, kPermute0Y, kPermute1Z, kPermute0X>(D1, D2);

  C0 = VectorNegativeMultiplySubtract(V0[0], V1[0], C0);
  C2 = VectorNegativeMultiplySubtract(V0[1], V1[1], C2);
  C4 = VectorNegativeMultiplySubtract(V0[2], V1[2], C4);
  C6 = VectorNegativeMultiplySubtract(V0[3], V1[3], C6);

  V0[0] = VectorSwizzle<kSwizzleW, kSwizzleX, kSwizzleW, kSwizzleX>(mt.c[1]);
  V1[0] = VectorPermute<kPermute0Z, kPermute1Y, kPermute1X, kPermute0Z>(D0, D2);
  V0[1] = VectorSwizzle<kSwizzleY, kSwizzleW, kSwizzleX, kSwizzleZ>(mt.c[0]);
  V1[1] = VectorPermute<kPermute1Y, kPermute0X, kPermute0W, kPermute1X>(D0, D2);
  V0[2] = VectorSwizzle<kSwizzleW, kSwizzleX, kSwizzleW, kSwizzleX>(mt.c[3]);
  V1[2] = VectorPermute<kPermute0Z, kPermute1W, kPermute1Z, kPermute0Z>(D1, D2);
  V0[3] = VectorSwizzle<kSwizzleY, kSwizzleW, kSwizzleX, kSwizzleZ>(mt.c[2]);
  V1[3] = VectorPermute<kPermute1W, kPermute0X, kPermute0W, kPermute1Z>(D1, D2);

  Vector C1 = VectorNegativeMultiplySubtract(V0[0], V1[0], C0);
  C0 = VectorMultiplyAdd(V0[0], V1[0], C0);
  Vector C3 = VectorMultiplyAdd(V0[1], V1[1], C2);
  C2 = VectorNegativeMultiplySubtract(V0[1], V1[1], C2);
  Vector C5 = VectorNegativeMultiplySubtract(V0[2], V1[2], C4);
  C4 = VectorMultiplyAdd(V0[2], V1[2], C4);
  Vector C7 = VectorMultiplyAdd(V0[3], V1[3], C6);
  C6 = VectorNegativeMultiplySubtract(V0[3], V1[3], C6);

  Matrix R;
  R.c[0] = VectorSelect(C0, C1, kSelect0101.v);
  R.c[1] = VectorSelect(C2, C3, kSelect0101.v);
  R.c[2] = VectorSelect(C4, C5, kSelect0101.v);
  R.c[3] = VectorSelect(C6, C7, kSelect0101.v);

  Vector Determinant = Vector4Dot(R.c[0], mt.c[0]);

  if (determinant != nullptr)
    *determinant = Determinant;

  Vector Reciprocal = VectorReciprocal(Determinant);

  Matrix Result;
  Result.c[0] = VectorMultiply(R.c[0], Reciprocal);
  Result.c[1] = VectorMultiply(R.c[1], Reciprocal);
  Result.c[2] = VectorMultiply(R.c[2], Reciprocal);
  Result.c[3] = VectorMultiply(R.c[3], Reciprocal);
  return Result;
}
LLX_INLINE Vector MatrixDeterminant(FMatrix m) {
  static const VectorF32 Sign = {1.0f, -1.0f, 1.0f, -1.0f};

  Vector V0 = VectorSwizzle<kSwizzleY, kSwizzleX, kSwizzleX, kSwizzleX>(m.c[2]);
  Vector V1 = VectorSwizzle<kSwizzleZ, kSwizzleZ, kSwizzleY, kSwizzleY>(m.c[3]);
  Vector V2 = VectorSwizzle<kSwizzleY, kSwizzleX, kSwizzleX, kSwizzleX>(m.c[2]);
  Vector V3 = VectorSwizzle<kSwizzleW, kSwizzleW, kSwizzleW, kSwizzleZ>(m.c[3]);
  Vector V4 = VectorSwizzle<kSwizzleZ, kSwizzleZ, kSwizzleY, kSwizzleY>(m.c[2]);
  Vector V5 = VectorSwizzle<kSwizzleW, kSwizzleW, kSwizzleW, kSwizzleZ>(m.c[3]);

  Vector P0 = VectorMultiply(V0, V1);
  Vector P1 = VectorMultiply(V2, V3);
  Vector P2 = VectorMultiply(V4, V5);

  V0 = VectorSwizzle<kSwizzleZ, kSwizzleZ, kSwizzleY, kSwizzleY>(m.c[2]);
  V1 = VectorSwizzle<kSwizzleY, kSwizzleX, kSwizzleX, kSwizzleX>(m.c[3]);
  V2 = VectorSwizzle<kSwizzleW, kSwizzleW, kSwizzleW, kSwizzleZ>(m.c[2]);
  V3 = VectorSwizzle<kSwizzleY, kSwizzleX, kSwizzleX, kSwizzleX>(m.c[3]);
  V4 = VectorSwizzle<kSwizzleW, kSwizzleW, kSwizzleW, kSwizzleZ>(m.c[2]);
  V5 = VectorSwizzle<kSwizzleZ, kSwizzleZ, kSwizzleY, kSwizzleY>(m.c[3]);

  P0 = VectorNegativeMultiplySubtract(V0, V1, P0);
  P1 = VectorNegativeMultiplySubtract(V2, V3, P1);
  P2 = VectorNegativeMultiplySubtract(V4, V5, P2);

  V0 = VectorSwizzle<kSwizzleW, kSwizzleW, kSwizzleW, kSwizzleZ>(m.c[1]);
  V1 = VectorSwizzle<kSwizzleZ, kSwizzleZ, kSwizzleY, kSwizzleY>(m.c[1]);
  V2 = VectorSwizzle<kSwizzleY, kSwizzleX, kSwizzleX, kSwizzleX>(m.c[1]);

  Vector S = VectorMultiply(m.c[0], Sign.v);
  Vector R = VectorMultiply(V0, P0);
  R = VectorNegativeMultiplySubtract(V1, P1, R);
  R = VectorMultiplyAdd(V2, P2, R);

  return Vector4Dot(S, R);
}
LLX_INLINE Matrix MatrixIdentity() {
  Matrix M;
  M.c[0] = kIdentityX.v;
  M.c[1] = kIdentityY.v;
  M.c[2] = kIdentityZ.v;
  M.c[3] = kIdentityW.v;
  return M;
}

LLX_INLINE Matrix MatrixTranslate(float offset_x, float offset_y,
                                  float offset_z) {
  Matrix r;
  r.c[0] = kIdentityX.v;
  r.c[1] = kIdentityY.v;
  r.c[2] = kIdentityZ.v;
  r.c[3] = VectorSet(offset_x, offset_y, offset_z, 1.f);
  return r;
}

LLX_INLINE Matrix MatrixTranslate(FVector offset) {
  Matrix r;
  r.c[0] = kIdentityX.v;
  r.c[1] = kIdentityY.v;
  r.c[2] = kIdentityZ.v;
  r.c[3] = VectorSelect(kIdentityW.v, offset, kSelect1110);
  return r;
}

LLX_INLINE Matrix MatrixScale(float scale_x, float scale_y, float scale_z) {
  Matrix r;
#if USE_ARM_NEON
  const static Vector s_zero = vdupq_n_f32(0);
  r.c[0] = vsetq_lane_f32(scale_x, s_zero, 0);
  r.c[1] = vsetq_lane_f32(scale_y, s_zero, 1);
  r.c[2] = vsetq_lane_f32(scale_z, s_zero, 2);
  r.c[3] = kIdentityW.v;
#else
  r._00 = scale_x;
  r._10 = 0.f;
  r._20 = 0.f;
  r._30 = 0.f;
  
  r._01 = 0.f;
  r._11 = scale_y;
  r._21 = 0.f;
  r._31 = 0.f;
  
  r._02 = 0.f;
  r._12 = 0.f;
  r._22 = scale_z;
  r._32 = 0.f;

  r._03 = 0.f;
  r._13 = 0.f;
  r._23 = 0.f;
  r._33 = 1.f;
#endif
  return r;
}

LLX_INLINE Matrix MatrixScale(FVector scale) {
  Matrix r;
#if USE_ARM_NEON
  r.c[0] = vandq_u32(scale, kMaskX);
  r.c[1] = vandq_u32(scale, kMaskY);
  r.c[2] = vandq_u32(scale, kMaskZ);
  r.c[3] = kIdentityW.v;
#else
  r._00 = scale.f32[0];
  r._10 = 0.f;
  r._20 = 0.f;
  r._30 = 0.f;
  
  r._01 = 0.f;
  r._11 = scale.f32[1];
  r._21 = 0.f;
  r._31 = 0.f;
  
  r._02 = 0.f;
  r._12 = 0.f;
  r._22 = scale.f32[2];
  r._32 = 0.f;
  
  r._03 = 0.f;
  r._13 = 0.f;
  r._23 = 0.f;
  r._33 = 1.f;
#endif
  return r;
}

LLX_INLINE Matrix MatrixRotateQuaternion(FVector quaternion) {
  Matrix r;
  return r;
}

// 旧坐标系到新坐标系变换顺序: 先平移到原坐标系原点,再旋转到新坐标
// ( R  0 )   ( I  -T )   ( R  -RT)
// ( 0  1 ) * ( 0   1 ) = ( 0    1)
//
// 平移矩阵T, 由translate构造
// 旋转矩阵R, 为(base1, base2, base3)构成矩阵的逆
// base1, base2, base3为新坐标系的正交基
LLX_INLINE Matrix MatrixSpaceTransform(FVector base_x, FVector base_y,
                                       FVector base_z, GVector translate) {
  // 计算 -T
  Vector neg_trans = VectorNegate(translate);
  Matrix r;

  // 计算 -RT
  Vector d0 = Vector3Dot(base_x, neg_trans);
  Vector d1 = Vector3Dot(base_y, neg_trans);
  Vector d2 = Vector3Dot(base_z, neg_trans);

  // ( R'  0)
  // (-RT  1)
  r.c[0] = VectorSelect(d0, base_x, kSelect1110);
  r.c[1] = VectorSelect(d1, base_y, kSelect1110);;
  r.c[2] = VectorSelect(d2, base_z, kSelect1110);;
  r.c[3] = kIdentityW.v;
  // 转置得到结果
  r = MatrixTranspose(r);
  return r;
}

LLX_INLINE Matrix MatrixLookAt(FVector look_pos, FVector eye_pos,
                               FVector up_dir) {
  return MatrixLookTo(VectorSubtract(look_pos, eye_pos), eye_pos, up_dir);
}

LLX_INLINE Matrix MatrixLookAt(FVector look_pos) {
#if USE_ARM_NEON
  const static Vector eye_pos = vdupq_n_f32(0);
#else
  const static Vector eye_pos = VectorSet(0.f, 0.f, 0.f, 0.f);
#endif
  return MatrixLookAt(look_pos, eye_pos, kIdentityY.v);
}

// look_dir作为Z轴
// up_dir作为Y轴
LLX_INLINE Matrix MatrixLookTo(FVector look_dir, FVector eye_pos,
                               FVector up_dir) {
  // Z轴
  Vector axis_z = Vector3Normalize(look_dir);
  // X = Cross(Y, Z)
  Vector axis_x = Vector3Normalize(Vector3Cross(up_dir, axis_z));
  // Y = Cross(Z, X), 正交化
  Vector axis_y = Vector3Cross(axis_z, axis_x);

  return MatrixSpaceTransform(axis_x, axis_y, axis_z, eye_pos);
}

LLX_INLINE Matrix MatrixLookTo(FVector look_dir) {
#if USE_ARM_NEON
  const static Vector eye_pos = vdupq_n_f32(0);
#else
  const static Vector eye_pos = VectorSet(0.f, 0.f, 0.f, 0.f);
#endif
  return MatrixLookTo(look_dir, eye_pos, kIdentityY.v);
}

// return m * v
LLX_INLINE Vector VectorTransform(FMatrix m, FVector v) {
  Vector r;
#if USE_ARM_NEON
  // (v0, v1)
  float32x2_t VL = vget_low_f32(v);
  // (v2, v3)
  float32x2_t VH = vget_high_f32(v);
  // vX = (m00 * v0, m10 * v0, m20 * v0, m30 * v0)
  Vector vX = vmulq_lane_f32(m.c[0], VL, 0);
  // vY = (m01 * v1, m11 * v1, m21 * v0, m31 * v1)
  Vector vY = vmulq_lane_f32(m.c[1], VL, 1);
  // vZ = (m02 * v2, m12 * v2, m22 * v2, m32 * v2) + vX
  Vector vZ = vmlaq_lane_f32(vX, m.c[2], VH, 0);
  // vW = (m03 * v3, m13 * v3, m23 * v3, m33 * v3) + vY
  Vector vW = vmlaq_lane_f32(vY, m.c[3], VH, 1);
  // r = (dot(r[0], v), dot(r[1], v), dot(r[2], v), dot(r[3], v))
  r = vaddq_f32(vZ, vW);
#else
  r.f32[0] = (m._00 * v.f32[0]) + (m._01 * v.f32[1]) +
             (m._02 * v.f32[2]) + (m._03 * v.f32[3]);
  r.f32[1] = (m._10 * v.f32[0]) + (m._11 * v.f32[1]) +
             (m._12 * v.f32[2]) + (m._13 * v.f32[3]);
  r.f32[2] = (m._20 * v.f32[0]) + (m._21 * v.f32[1]) +
             (m._22 * v.f32[2]) + (m._23 * v.f32[3]);
  r.f32[3] = (m._30 * v.f32[0]) + (m._31 * v.f32[1]) +
             (m._32 * v.f32[2]) + (m._33 * v.f32[3]);
#endif
  return r;
}

// ( a          )
// (    b       )
// (       c  d ) = P
// (       1    )
// Pv = zv'
// 其中:
// a = 1 / (aspect_ratio * tan(fov_y / 2))
// b = 1 / tan(fov_y / 2) = cos(fov_y / 2) / sin(fov_y / 2)
// 由上方乘法得到 z*z' = cz + d
// 且z = near_z时, z' = 0; z = far_z时, z' = 1
// 有c * near_z + d = 0
//   c * far_z + d = 1
// 则c = far_z / (far_z - near_z)
//   d = - near_z * far_z / (far_z - near_z)
LLX_INLINE Matrix MatrixPerspective(float fov_y, float aspect_ratio,
                                    float near_z, float far_z) {
  float sin_fov_y_2, cos_fov_y_2;
  ScalarSinCos(&sin_fov_y_2, &cos_fov_y_2, 0.5f * fov_y);
  float b = cos_fov_y_2 / sin_fov_y_2;
  float a = b / aspect_ratio;
  float c = far_z / (far_z - near_z);
  float d = -1 * near_z * c;

  Matrix r;
#if USE_ARM_NEON
  r.c[0] = vsetq_lane_f32(a, kZeroVector.v, 0);
  r.c[1] = vsetq_lane_f32(b, kZeroVector.v, 1);
  r.c[2] = vsetq_lane_f32(c, kIdentityW.v, 2);
  r.c[3] = vsetq_lane_f32(d, kZeroVector.v, 2);
#else
  memset(&r, 0, sizeof(Matrix));
  r._00 = a;
  r._11 = b;
  r._22 = c;
  r._23 = d;
  r._32 = 1;
#endif
  return r;
}
}
}
#endif // _LLX_UTIL_MATRIX_MATRICES_INLINE_H_