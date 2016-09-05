//
// Created by Marco.LIU on 16/8/18.
// Copyright (c) 2016 LLX. All rights reserved.
//

#ifndef _LLX_UTIL_MATRIX_VECTORS_INLINE_H_
#define _LLX_UTIL_MATRIX_VECTORS_INLINE_H_

#include "matrix_config.h"

#if NO_INLINE
#include "vectors.h"
#include "global_consts.h"
#endif

#include <assert.h>

#if !USE_ARM_NEON
#define LLX_IS_NAN(x)  ((*(uint32_t*)&(x) & 0x7F800000) == 0x7F800000 && \
                        (*(uint32_t*)&(x) & 0x7FFFFF) != 0)
#define LLX_IS_INF(x)  ((*(uint32_t*)&(x) & 0x7FFFFFFF) == 0x7F800000)
#endif

namespace LLX {
namespace util {
LLX_INLINE float ScalarSin(float radian) {
  // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
  float quotient = (float)LLX_1_2PI * radian;
  if (radian >= 0.0f) {
    quotient = (float)((int)(quotient + 0.5f));
  } else {
    quotient = (float)((int)(quotient - 0.5f));
  }
  float y = radian - (float)LLX_2PI * quotient;

  // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
  if (y > (float)LLX_PI_2) {
    y = (float)LLX_PI - y;
  } else if (y < (float)-LLX_PI_2) {
    y = (float)-LLX_PI - y;
  }

  // 11-degree minimax approximation
  float y2 = y * y;
  return (((((-2.3889859e-08f * y2 + 2.7525562e-06f) * y2 - 0.00019840874f) * y2
      + 0.0083333310f) * y2 - 0.16666667f) * y2 + 1.0f) * y;
}

LLX_INLINE float ScalarSinEst(float radian) {
  // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
  float quotient = (float)LLX_1_2PI * radian;
  if (radian >= 0.0f) {
    quotient = (float)((int)(quotient + 0.5f));
  } else {
    quotient = (float)((int)(quotient - 0.5f));
  }
  float y = radian - (float)LLX_2PI * quotient;

  // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
  if (y > (float)LLX_PI_2) {
    y = (float)LLX_PI - y;
  } else if (y < (float)-LLX_PI_2) {
    y = (float)-LLX_PI - y;
  }

  // 7-degree minimax approximation
  float y2 = y * y;
  return (((-0.00018524670f * y2 + 0.0083139502f) * y2 - 0.16665852f) * y2
      + 1.0f) * y;
}

LLX_INLINE float ScalarCos(float radian) {
  // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
  float quotient = (float)LLX_1_2PI * radian;
  if (radian >= 0.0f) {
    quotient = (float)((int)(quotient + 0.5f));
  } else {
    quotient = (float)((int)(quotient - 0.5f));
  }
  float y = radian - (float)LLX_2PI * quotient;

  // Map y to [-pi/2,pi/2] with cos(y) = sign*cos(x).
  float sign;
  if (y > (float)LLX_PI_2) {
    y = (float)LLX_PI - y;
    sign = -1.0f;
  } else if (y < (float)-LLX_PI_2) {
    y = (float)-LLX_PI - y;
    sign = -1.0f;
  } else {
    sign = +1.0f;
  }

  // 10-degree minimax approximation
  float y2 = y * y;
  float p = ((((-2.6051615e-07f * y2 + 2.4760495e-05f) * y2 - 0.0013888378f)
      * y2 + 0.041666638f) * y2 - 0.5f) * y2 + 1.0f;
  return sign * p;
}

LLX_INLINE float ScalarCosEst(float radian) {
  // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
  float quotient = (float)LLX_1_2PI * radian;
  if (radian >= 0.0f) {
    quotient = (float)((int)(quotient + 0.5f));
  } else {
    quotient = (float)((int)(quotient - 0.5f));
  }
  float y = radian - (float)LLX_2PI * quotient;

  // Map y to [-pi/2,pi/2] with cos(y) = sign*cos(x).
  float sign;
  if (y > (float)LLX_PI_2) {
    y = (float)LLX_PI - y;
    sign = -1.0f;
  } else if (y < (float)-LLX_PI_2) {
    y = (float)-LLX_PI - y;
    sign = -1.0f;
  } else {
    sign = +1.0f;
  }

  // 6-degree minimax approximation
  float y2 = y * y;
  float p =
      ((-0.0012712436f * y2 + 0.041493919f) * y2 - 0.49992746f) * y2 + 1.0f;
  return sign * p;
}

LLX_INLINE void ScalarSinCos(float* sine, float* cosine, float radian) {
  assert(sine);
  assert(cosine);

  // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
  float quotient = (float)LLX_1_2PI * radian;
  if (radian >= 0.0f) {
    quotient = (float)((int)(quotient + 0.5f));
  } else {
    quotient = (float)((int)(quotient - 0.5f));
  }
  float y = radian - (float)LLX_2PI * quotient;

  // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
  float sign;
  if (y > (float)LLX_PI_2) {
    y = (float)LLX_PI - y;
    sign = -1.0f;
  } else if (y < (float)-LLX_PI_2) {
    y = (float)-LLX_PI - y;
    sign = -1.0f;
  } else {
    sign = +1.0f;
  }

  float y2 = y * y;

  // 11-degree minimax approximation
  *sine = (((((-2.3889859e-08f * y2 + 2.7525562e-06f) * y2 - 0.00019840874f)
      * y2 + 0.0083333310f) * y2 - 0.16666667f) * y2 + 1.0f) * y;

  // 10-degree minimax approximation
  float p = ((((-2.6051615e-07f * y2 + 2.4760495e-05f) * y2 - 0.0013888378f)
      * y2 + 0.041666638f) * y2 - 0.5f) * y2 + 1.0f;
  *cosine = sign * p;
}

LLX_INLINE void ScalarSinCosEst(float* sine, float* cosine, float radian) {
  assert(sine);
  assert(cosine);

  // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
  float quotient = (float)LLX_1_2PI * radian;
  if (radian >= 0.0f) {
    quotient = (float)((int)(quotient + 0.5f));
  } else {
    quotient = (float)((int)(quotient - 0.5f));
  }
  float y = radian - (float)LLX_2PI * quotient;

  // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
  float sign;
  if (y > (float)LLX_PI_2) {
    y = (float)LLX_PI - y;
    sign = -1.0f;
  } else if (y < (float)-LLX_PI_2) {
    y = (float)-LLX_PI - y;
    sign = -1.0f;
  } else {
    sign = +1.0f;
  }

  float y2 = y * y;

  // 7-degree minimax approximation
  *sine = (((-0.00018524670f * y2 + 0.0083139502f) * y2 - 0.16665852f)
      * y2 + 1.0f) * y;

  // 6-degree minimax approximation
  float p =
      ((-0.0012712436f * y2 + 0.041493919f) * y2 - 0.49992746f) * y2 + 1.0f;
  *cosine = sign * p;
}

LLX_INLINE float ScalarASin(float sine) {
  // Clamp input to [-1,1].
  bool nonnegative = (sine >= 0.0f);
  float x = fabsf(sine);
  float omx = 1.0f - x;
  if (omx < 0.0f) {
    omx = 0.0f;
  }
  float root = sqrtf(omx);

  // 7-degree minimax approximation
  float result = ((((((-0.0012624911f * x + 0.0066700901f) * x - 0.0170881256f)
      * x + 0.0308918810f) * x - 0.0501743046f) * x + 0.0889789874f) * x
      - 0.2145988016f) * x + 1.5707963050f;
  result *= root;  // acos(|x|)

  // acos(x) = pi - acos(-x) when x < 0, asin(x) = pi/2 - acos(x)
  return (nonnegative ? (float)LLX_PI_2 - result : result - (float)LLX_PI_2);
}

LLX_INLINE float ScalarASinEst(float sine) {
  // Clamp input to [-1,1].
  bool nonnegative = (sine >= 0.0f);
  float x = fabsf(sine);
  float omx = 1.0f - x;
  if (omx < 0.0f) {
    omx = 0.0f;
  }
  float root = sqrtf(omx);

  // 3-degree minimax approximation
  float result =
      ((-0.0187293f * x + 0.0742610f) * x - 0.2121144f) * x + 1.5707288f;
  result *= root;  // acos(|x|)

  // acos(x) = pi - acos(-x) when x < 0, asin(x) = pi/2 - acos(x)
  return (nonnegative ? (float)LLX_PI_2 - result : result - (float)LLX_PI_2);
}

LLX_INLINE float ScalarACos(float cosine) {
  // Clamp input to [-1,1].
  bool nonnegative = (cosine >= 0.0f);
  float x = fabsf(cosine);
  float omx = 1.0f - x;
  if (omx < 0.0f) {
    omx = 0.0f;
  }
  float root = sqrtf(omx);

  // 7-degree minimax approximation
  float result = ((((((-0.0012624911f * x + 0.0066700901f) * x - 0.0170881256f)
      * x + 0.0308918810f) * x - 0.0501743046f) * x + 0.0889789874f) * x
      - 0.2145988016f) * x + 1.5707963050f;
  result *= root;

  // acos(x) = pi - acos(-x) when x < 0
  return (nonnegative ? result : (float)LLX_PI - result);
}

LLX_INLINE float ScalarACosEst(float cosine) {
  // Clamp input to [-1,1].
  bool nonnegative = (cosine >= 0.0f);
  float x = fabsf(cosine);
  float omx = 1.0f - x;
  if (omx < 0.0f) {
    omx = 0.0f;
  }
  float root = sqrtf(omx);

  // 3-degree minimax approximation
  float result =
      ((-0.0187293f * x + 0.0742610f) * x - 0.2121144f) * x + 1.5707288f;
  result *= root;

  // acos(x) = pi - acos(-x) when x < 0
  return (nonnegative ? result : (float)LLX_PI - result);
}


LLX_INLINE Vector VectorZero() {
#if USE_ARM_NEON
  return vdupq_n_f32(0);
#else
  Vector r = {0.0f,0.0f,0.0f,0.0f};
  return r;
#endif
}
LLX_INLINE Vector VectorSet(float x, float y, float z, float w) {
#if USE_ARM_NEON
  float32x2_t V0 = vcreate_f32(((uint64_t)*(const uint32_t *)&x) |
                               ((uint64_t)(*(const uint32_t *)&y) << 32));
  float32x2_t V1 = vcreate_f32(((uint64_t)*(const uint32_t *)&z) |
                               ((uint64_t)(*(const uint32_t *)&w) << 32));
  return vcombine_f32(V0, V1);
#else
  Vector r = {w, y, z, w};
  return r;
#endif
}
LLX_INLINE Vector VectorReplicate(float val) {
#if USE_ARM_NEON
  return vdupq_n_f32(val);
#else
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = val;
  return r;
#endif
}
LLX_INLINE Vector VectorReplicate(const float* val) {
  assert(val);
#if USE_ARM_NEON
  return vld1q_dup_f32(val);
#else
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = *val;
  return r;
#endif
}
LLX_INLINE Vector VectorReplicate(uint32_t val) {
#if USE_ARM_NEON
  return vdupq_n_u32(val);
#else
  Vector r;
  r.u32[0] = r.u32[1] = r.u32[2] = r.u32[3] = val;
  return r;
#endif
}
LLX_INLINE Vector VectorReplicate(const uint32_t* val) {
#if USE_ARM_NEON
  return vld1q_dup_u32(val);
#else
  Vector r;
  r.u32[0] = r.u32[1] = r.u32[2] = r.u32[3] = *val;
  return r;
#endif
}
LLX_INLINE Vector VectorSplatX(FVector v) {
#if USE_ARM_NEON
  return vdupq_lane_f32(vget_low_f32(v), 0);
#else
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = v.f32[0];
  return r;
#endif
}
LLX_INLINE Vector VectorSplatY(FVector v) {
#if USE_ARM_NEON
  return vdupq_lane_f32(vget_low_f32(v), 1);
#else
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = v.f32[1];
  return r;
#endif
}
LLX_INLINE Vector VectorSplatZ(FVector v) {
#if USE_ARM_NEON
  return vdupq_lane_f32(vget_high_f32(v), 0);
#else
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = v.f32[2];
  return r;
#endif
}
LLX_INLINE Vector VectorSplatW(FVector v) {
#if USE_ARM_NEON
  return vdupq_lane_f32(vget_high_f32(v), 1);
#else
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = v.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorSplatOne() {
#if USE_ARM_NEON
  return vdupq_n_f32(1.0f);
#else
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = 1.f;
  return r;
#endif
}
LLX_INLINE Vector VectorSplatInfinit() {
#if USE_ARM_NEON
  return vdupq_n_u32(0x7F800000);
#else
  Vector r;
  r.u32[0] = r.u32[1] = r.u32[2] = r.u32[3] = 0x7F800000;
  return r;
#endif
}
LLX_INLINE Vector VectorSplatQNaN() {
#if USE_ARM_NEON
  return vdupq_n_u32(0x7FC00000);
#else
  Vector r;
  r.u32[0] = r.u32[1] = r.u32[2] = r.u32[3] = 0x7FC00000;
  return r;
#endif
}
LLX_INLINE Vector VectorSplatEpsilon() {
#if USE_ARM_NEON
  return vdupq_n_u32(0x34000000);
#else
  Vector r;
  r.u32[0] = r.u32[1] = r.u32[2] = r.u32[3] = 0x34000000;
  return r;
#endif
}
LLX_INLINE Vector VectorSplatSignMask() {
#if USE_ARM_NEON
  return vdupq_n_u32(0x80000000U);
#else
  Vector r;
  r.u32[0] = r.u32[1] = r.u32[2] = r.u32[3] = 0x80000000U;
  return r;
#endif
}

/*LLX_INLINE float VectorGetByIndex(FVector v, size_t i) {
  assert(i < 4);
#if USE_ARM_NEON
  return vgetq_lane_f32(v, i);
#else
  return v.f32[i];
#endif
}*/
LLX_INLINE float VectorGetX(FVector v) {
#if USE_ARM_NEON
  return vgetq_lane_f32(v, 0);
#else
  return v.f32[0];
#endif
}
LLX_INLINE float VectorGetY(FVector v) {
#if USE_ARM_NEON
  return vgetq_lane_f32(v, 1);
#else
  return v.f32[1];
#endif
}
LLX_INLINE float VectorGetZ(FVector v) {
#if USE_ARM_NEON
  return vgetq_lane_f32(v, 2);
#else
  return v.f32[2];
#endif
}
LLX_INLINE float VectorGetW(FVector v) {
#if USE_ARM_NEON
  return vgetq_lane_f32(v, 3);
#else
  return v.f32[3];
#endif
}

/*LLX_INLINE void VectorGetByIndex(float* f, FVector v, size_t i) {
  assert(i < 4);
  assert(f != nullptr);
#if USE_ARM_NEON
  vst1q_lane_f32(f, v, i);
#else
  *f = v.f32[i];
#endif
}*/
LLX_INLINE void VectorGetX(float* f, FVector v) {
  assert(f != nullptr);
#if USE_ARM_NEON
  vst1q_lane_f32(f, v, 0);
#else
  *f = v.f32[0];
#endif
}
LLX_INLINE void VectorGetY(float* f, FVector v) {
  assert(f != nullptr);
#if USE_ARM_NEON
  vst1q_lane_f32(f, v, 1);
#else
  *f = v.f32[1];
#endif
}
LLX_INLINE void VectorGetZ(float* f, FVector v) {
  assert(f != nullptr);
#if USE_ARM_NEON
  vst1q_lane_f32(f, v, 2);
#else
  *f = v.f32[2];
#endif
}
LLX_INLINE void VectorGetW(float* f, FVector v) {
  assert(f != nullptr);
#if USE_ARM_NEON
  vst1q_lane_f32(f, v, 3);
#else
  *f = v.f32[3];
#endif
}

/*LLX_INLINE Vector VectorSetByIndex(FVector v, float f, size_t i) {
  assert(i < 4);
#if USE_ARM_NEON
  return vsetq_lane_f32(f, v, i);
#else
  Vector r = v;
  r.f32[i] = f;
  return r;
#endif
}*/
LLX_INLINE Vector VectorSetX(FVector v, float f) {
#if USE_ARM_NEON
  return vsetq_lane_f32(f, v, 0);
#else
  Vector r = v;
  r.f32[0] = f;
  return r;
#endif
}
LLX_INLINE Vector VectorSetY(FVector v, float f) {
#if USE_ARM_NEON
  return vsetq_lane_f32(f, v, 1);
#else
  Vector r = v;
  r.f32[1] = f;
  return r;
#endif
}
LLX_INLINE Vector VectorSetZ(FVector v, float f) {
#if USE_ARM_NEON
  return vsetq_lane_f32(f, v, 2);
#else
  Vector r = v;
  r.f32[2] = f;
  return r;
#endif
}
LLX_INLINE Vector VectorSetW(FVector v, float f) {
#if USE_ARM_NEON
  return vsetq_lane_f32(f, v, 3);
#else
  Vector r = v;
  r.f32[3] = f;
  return r;
#endif
}

/*LLX_INLINE Vector VectorSetByIndex(FVector v, const float* f, size_t i) {
  assert(i < 4);
  assert(f != nullptr);
#if USE_ARM_NEON
  return vld1q_lane_f32(f, v, i);
#else
  Vector r = v;
  r.f32[i] = *f;
  return r;
#endif
}*/
LLX_INLINE Vector VectorSetX(FVector v, const float* f) {
#if USE_ARM_NEON
  return vld1q_lane_f32(f, v, 0);
#else
  Vector r = v;
  r.f32[0] = *f;
  return r;
#endif
}
LLX_INLINE Vector VectorSetY(FVector v, const float* f) {
#if USE_ARM_NEON
  return vld1q_lane_f32(f, v, 1);
#else
  Vector r = v;
  r.f32[1] = *f;
  return r;
#endif
}
LLX_INLINE Vector VectorSetZ(FVector v, const float* f) {
#if USE_ARM_NEON
  return vld1q_lane_f32(f, v, 2);
#else
  Vector r = v;
  r.f32[2] = *f;
  return r;
#endif
}
LLX_INLINE Vector VectorSetW(FVector v, const float* f) {
#if USE_ARM_NEON
  return vld1q_lane_f32(f, v, 3);
#else
  Vector r = v;
  r.f32[3] = *f;
  return r;
#endif
}

LLX_INLINE Vector VectorSwizzle(FVector v, uint32_t e0, uint32_t e1,
                                uint32_t e2, uint32_t e3) {
  assert((e0 < 4) && (e1 < 4) && (e2 < 4) && (e3 < 4));
#if USE_ARM_NEON
  static const uint32_t ctrl_ele[4] =
  {
      0x03020100, // SWIZZLE_X
      0x07060504, // SWIZZLE_Y
      0x0B0A0908, // SWIZZLE_Z
      0x0F0E0D0C, // SWIZZLE_W
  };

  uint8x8x2_t tbl;
  tbl.val[0] = vget_low_f32(v);
  tbl.val[1] = vget_high_f32(v);

  uint32x2_t idx = vcreate_u32(((uint64_t)ctrl_ele[e0]) |
                               (((uint64_t)ctrl_ele[e1]) << 32));
  const uint8x8_t rL = vtbl2_u8(tbl, idx);

  idx = vcreate_u32(((uint64_t)ctrl_ele[e2]) |
                    (((uint64_t)ctrl_ele[e3]) << 32));
  const uint8x8_t rH = vtbl2_u8(tbl, idx);

  return vcombine_f32(rL, rH);
#else
  Vector r = { v.f32[e0], v.f32[e1], v.f32[e2], v.f32[e3] };
  return r;
#endif
}

LLX_INLINE Vector VectorPermute(FVector v1, FVector v2, uint32_t permute_x,
                                uint32_t permute_y, uint32_t permute_z,
                                uint32_t permute_w) {
  assert(permute_x <= 7 && permute_y <= 7 && permute_z <= 7 && permute_w <= 7);
#if USE_ARM_NEON
  static const uint32_t ctrl_ele[8] =
  {
      0x03020100, // PERMUTE_0X
      0x07060504, // PERMUTE_0Y
      0x0B0A0908, // PERMUTE_0Z
      0x0F0E0D0C, // PERMUTE_0W
      0x13121110, // PERMUTE_1X
      0x17161514, // PERMUTE_1Y
      0x1B1A1918, // PERMUTE_1Z
      0x1F1E1D1C, // PERMUTE_1W
  };

  uint8x8x4_t tbl;
  tbl.val[0] = vget_low_f32(v1);
  tbl.val[1] = vget_high_f32(v1);
  tbl.val[2] = vget_low_f32(v2);
  tbl.val[3] = vget_high_f32(v2);

  uint32x2_t idx = vcreate_u32(((uint64_t)ctrl_ele[permute_x]) |
                               (((uint64_t)ctrl_ele[permute_y]) << 32));
  const uint8x8_t rL = vtbl4_u8(tbl, idx);

  idx = vcreate_u32(((uint64_t)ctrl_ele[permute_z]) |
                    (((uint64_t)ctrl_ele[permute_w]) << 32));
  const uint8x8_t rH = vtbl4_u8(tbl, idx);

  return vcombine_f32(rL, rH);
#else
  const uint32_t* aPtr[2];
  aPtr[0] = (const uint32_t*)(&v1);
  aPtr[1] = (const uint32_t*)(&v2);

  Vector r;
  uint32_t* pWork = (uint32_t*)(&r);

  const uint32_t i0 = permute_x & 3;
  const uint32_t vi0 = permute_x >> 2;
  pWork[0] = aPtr[vi0][i0];

  const uint32_t i1 = permute_y & 3;
  const uint32_t vi1 = permute_y >> 2;
  pWork[1] = aPtr[vi1][i1];

  const uint32_t i2 = permute_z & 3;
  const uint32_t vi2 = permute_z >> 2;
  pWork[2] = aPtr[vi2][i2];

  const uint32_t i3 = permute_w & 3;
  const uint32_t vi3 = permute_w >> 2;
  pWork[3] = aPtr[vi3][i3];

  return r;
#endif
}
LLX_INLINE Vector VectorSelectControl(uint32_t index0, uint32_t index1,
                                      uint32_t index2, uint32_t index3) {
  assert((index0 < 2) && (index1 < 2) && (index2 < 2) && (index3 < 2));
#if USE_ARM_NEON
  int32x2_t v0 = vcreate_s32(((uint64_t)index0) | ((uint64_t)index1 << 32));
  int32x2_t v1 = vcreate_s32(((uint64_t)index2) | ((uint64_t)index3 << 32));
  int32x4_t tmp = vcombine_s32(v0, v1);
  // Any non-zero entries become 0xFFFFFFFF else 0
  return vcgtq_s32(tmp, kZeroVector);
#else
  Vector r;
  const uint32_t ctrl_ele[] = {kSelect0, kSelect1};
  r.u32[0] = ctrl_ele[index0];
  r.u32[1] = ctrl_ele[index1];
  r.u32[2] = ctrl_ele[index2];
  r.u32[3] = ctrl_ele[index3];
  return r;
#endif
}
LLX_INLINE Vector VectorSelect(FVector v1, FVector v2, FVector control) {
#if USE_ARM_NEON
  return vbslq_f32(control, v2, v1);
#else
  Vector r;
  r.u32[0] = (v1.u32[0] & ~control.u32[0]) | (v2.u32[0] & control.u32[0]);
  r.u32[1] = (v1.u32[1] & ~control.u32[1]) | (v2.u32[1] & control.u32[1]);
  r.u32[2] = (v1.u32[2] & ~control.u32[2]) | (v2.u32[2] & control.u32[2]);
  r.u32[3] = (v1.u32[3] & ~control.u32[3]) | (v2.u32[3] & control.u32[3]);
  return r;
#endif
}
LLX_INLINE Vector VectorMergeXY(FVector v1, FVector v2) {
#if USE_ARM_NEON
  return vzipq_f32(v1, v2).val[0];
#else
  Vector r;
  r.u32[0] = v1.u32[0];
  r.u32[1] = v2.u32[0];
  r.u32[2] = v1.u32[1];
  r.u32[3] = v2.u32[1];
  return r;
#endif
}
LLX_INLINE Vector VectorMergeZW(FVector v1, FVector v2) {
#if USE_ARM_NEON
  return vzipq_f32(v1, v2).val[1];
#else
  Vector r;
  r.u32[0] = v1.u32[2];
  r.u32[1] = v2.u32[2];
  r.u32[2] = v1.u32[3];
  r.u32[3] = v2.u32[3];
  return r;
#endif
}

LLX_INLINE Vector VectorEqual(FVector v1, FVector v2) {
#if USE_ARM_NEON
  return vceqq_f32(v1, v2);
#else
  Vector r;
  r.u32[0] = (v1.f32[0] == v2.f32[0]) ? 0xFFFFFFFF : 0;
  r.u32[1] = (v1.f32[1] == v2.f32[1]) ? 0xFFFFFFFF : 0;
  r.u32[2] = (v1.f32[2] == v2.f32[2]) ? 0xFFFFFFFF : 0;
  r.u32[3] = (v1.f32[3] == v2.f32[3]) ? 0xFFFFFFFF : 0;
  return r;
#endif
}
LLX_INLINE Vector VectorEqualInt(FVector v1, FVector v2) {
#if USE_ARM_NEON
  return vceqq_u32(v1, v2);
#else
  Vector r;
  r.u32[0] = (v1.u32[0] == v2.u32[0]) ? 0xFFFFFFFF : 0;
  r.u32[1] = (v1.u32[1] == v2.u32[1]) ? 0xFFFFFFFF : 0;
  r.u32[2] = (v1.u32[2] == v2.u32[2]) ? 0xFFFFFFFF : 0;
  r.u32[3] = (v1.u32[3] == v2.u32[3]) ? 0xFFFFFFFF : 0;
  return r;
#endif
}
LLX_INLINE Vector VectorNearEqual(FVector v1, FVector v2, FVector epsilon) {
#if USE_ARM_NEON
  Vector d = vsubq_f32(v1, v2);
  return vcleq_f32(d, epsilon);
#else
  float dx = fabsf(v1.f32[0] - v2.f32[0]);
  float dy = fabsf(v1.f32[1] - v2.f32[1]);
  float dz = fabsf(v1.f32[2] - v2.f32[2]);
  float dw = fabsf(v1.f32[3] - v2.f32[3]);

  Vector r;
  r.u32[0] = (dx <= epsilon.f32[0]) ? 0xFFFFFFFF : 0;
  r.u32[1] = (dy <= epsilon.f32[1]) ? 0xFFFFFFFF : 0;
  r.u32[2] = (dz <= epsilon.f32[2]) ? 0xFFFFFFFF : 0;
  r.u32[3] = (dw <= epsilon.f32[3]) ? 0xFFFFFFFF : 0;
  return r;
#endif
}
LLX_INLINE Vector VectorNotEqual(FVector v1, FVector v2) {
#if USE_ARM_NEON
  return vmvnq_u32(vceqq_f32(v1, v2));
#else
  Vector r;
  r.u32[0] = (v1.f32[0] != v2.f32[0]) ? 0xFFFFFFFF : 0;
  r.u32[1] = (v1.f32[1] != v2.f32[1]) ? 0xFFFFFFFF : 0;
  r.u32[2] = (v1.f32[2] != v2.f32[2]) ? 0xFFFFFFFF : 0;
  r.u32[3] = (v1.f32[3] != v2.f32[3]) ? 0xFFFFFFFF : 0;
  return r;
#endif
}
LLX_INLINE Vector VectorIsNaN(FVector v) {
#if USE_ARM_NEON
  // Test against itself. NaN is always not equal
  uint32x4_t tmp = vceqq_f32(v, v);
  // Flip results
  return vmvnq_u32(tmp);
#else
  Vector r;
  r.u32[0] = (LLX_IS_NAN(v.f32[0])) ? 0xFFFFFFFF : 0;
  r.u32[1] = (LLX_IS_NAN(v.f32[1])) ? 0xFFFFFFFF : 0;
  r.u32[2] = (LLX_IS_NAN(v.f32[2])) ? 0xFFFFFFFF : 0;
  r.u32[3] = (LLX_IS_NAN(v.f32[3])) ? 0xFFFFFFFF : 0;
  return r;
#endif
}
LLX_INLINE Vector VectorIsInfinite(FVector v) {
#if USE_ARM_NEON
  // Mask off the sign bit
  uint32x4_t tmp = vandq_u32(v, kAbsMask);
  // Compare to infinity
  tmp = vceqq_f32(tmp, kInfinity);
  // If any are infinity, the signs are true.
  return tmp;
#else
  Vector r;
  r.u32[0] = (LLX_IS_INF(v.f32[0])) ? 0xFFFFFFFF : 0;
  r.u32[1] = (LLX_IS_INF(v.f32[1])) ? 0xFFFFFFFF : 0;
  r.u32[2] = (LLX_IS_INF(v.f32[2])) ? 0xFFFFFFFF : 0;
  r.u32[3] = (LLX_IS_INF(v.f32[3])) ? 0xFFFFFFFF : 0;
  return r;
#endif
}
LLX_INLINE Vector VectorNegate(FVector v) {
#if USE_ARM_NEON
  return vnegq_f32(v);
#else
  Vector r;
  r.f32[0] = -v.f32[0];
  r.f32[1] = -v.f32[1];
  r.f32[2] = -v.f32[2];
  r.f32[3] = -v.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorAdd(FVector v1, FVector v2) {
#if USE_ARM_NEON
  return vaddq_f32(v1, v2);
#else
  Vector r;
  r.f32[0] = v1.f32[0] + v2.f32[0];
  r.f32[1] = v1.f32[1] + v2.f32[1];
  r.f32[2] = v1.f32[2] + v2.f32[2];
  r.f32[3] = v1.f32[3] + v2.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorSubtract(FVector v1, FVector v2) {
#if USE_ARM_NEON
  return vsubq_f32(v1, v2);
#else
  Vector r;
  r.f32[0] = v1.f32[0] - v2.f32[0];
  r.f32[1] = v1.f32[1] - v2.f32[1];
  r.f32[2] = v1.f32[2] - v2.f32[2];
  r.f32[3] = v1.f32[3] - v2.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorMultiply(FVector v1, FVector v2) {
#if USE_ARM_NEON
  return vmulq_f32(v1, v2);
#else
  Vector r;
  r.f32[0] = v1.f32[0] * v2.f32[0];
  r.f32[1] = v1.f32[1] * v2.f32[1];
  r.f32[2] = v1.f32[2] * v2.f32[2];
  r.f32[3] = v1.f32[3] * v2.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorMultiplyAdd(FVector v1, FVector v2, FVector v3) {
#if USE_ARM_NEON
  return vmlaq_f32(v3, v1, v2);
#else
  Vector r;
  r.f32[0] = v1.f32[0] * v2.f32[0] + v3.f32[0];
  r.f32[1] = v1.f32[1] * v2.f32[1] + v3.f32[1];
  r.f32[2] = v1.f32[2] * v2.f32[2] + v3.f32[2];
  r.f32[3] = v1.f32[3] * v2.f32[3] + v3.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorDivide(FVector v1, FVector v2) {
#if USE_ARM_NEON
  // 2 iterations of Newton-Raphson refinement of reciprocal
  float32x4_t reciprocal = vrecpeq_f32(v2);
  float32x4_t s = vrecpsq_f32(reciprocal, v2);
  reciprocal = vmulq_f32(s, reciprocal);
  s = vrecpsq_f32(reciprocal, v2);
  reciprocal = vmulq_f32(s, reciprocal);
  return vmulq_f32(v1, reciprocal);
#else
  Vector r;
  r.f32[0] = v1.f32[0] / v2.f32[0];
  r.f32[1] = v1.f32[1] / v2.f32[1];
  r.f32[2] = v1.f32[2] / v2.f32[2];
  r.f32[3] = v1.f32[3] / v2.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorScale(FVector v, float scale) {
#if USE_ARM_NEON
  return vmulq_n_f32(v, scale);
#else
  Vector r;
  r.f32[0] = v.f32[0] * scale;
  r.f32[1] = v.f32[1] * scale;
  r.f32[2] = v.f32[2] * scale;
  r.f32[3] = v.f32[3] * scale;
  return r;
#endif
}
LLX_INLINE Vector VectorNegativeMultiplySubtract(FVector v1, FVector v2,
                                                 FVector v3) {
#if USE_ARM_NEON
  return vmlsq_f32(v3, v1, v2);
#else
  Vector r;
  r.f32[0] = v3.f32[0] - (v1.f32[0] * v2.f32[0]);
  r.f32[1] = v3.f32[1] - (v1.f32[1] * v2.f32[1]);
  r.f32[2] = v3.f32[2] - (v1.f32[2] * v2.f32[2]);
  r.f32[3] = v3.f32[3] - (v1.f32[3] * v2.f32[3]);
  return r;
#endif
}
LLX_INLINE Vector VectorReciprocalEst(FVector v) {
#if USE_ARM_NEON
  return vrecpeq_f32(v);
#else
  Vector r;
  r.f32[0] = 1.f / v.f32[0];
  r.f32[1] = 1.f / v.f32[1];
  r.f32[2] = 1.f / v.f32[2];
  r.f32[3] = 1.f / v.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorReciprocal(FVector v) {
#if USE_ARM_NEON
  // 2 iterations of Newton-Raphson refinement
  float32x4_t reciprocal = vrecpeq_f32(v);
  float32x4_t s = vrecpsq_f32(reciprocal, v);
  reciprocal = vmulq_f32(s, reciprocal);
  s = vrecpsq_f32(reciprocal, v);
  return vmulq_f32(s, reciprocal);
#else
  Vector r;
  r.f32[0] = 1.f / v.f32[0];
  r.f32[1] = 1.f / v.f32[1];
  r.f32[2] = 1.f / v.f32[2];
  r.f32[3] = 1.f / v.f32[3];
  return r;
#endif
}
LLX_INLINE Vector VectorSqrtEst(FVector v) {
#if USE_ARM_NEON
  // 1 iteration of Newton-Raphson refinment of sqrt
  float32x4_t s0 = vrsqrteq_f32(v);
  float32x4_t p0 = vmulq_f32(v, s0);
  float32x4_t r0 = vrsqrtsq_f32(p0, s0);
  float32x4_t s1 = vmulq_f32(s0, r0);

  Vector is_infinity = VectorEqualInt(v, kInfinity.v);
  Vector is_zero = VectorEqual(v, vdupq_n_f32(0));
  Vector result = vmulq_f32(v, s1 );
  Vector select = VectorEqualInt(is_infinity, is_zero);
  return VectorSelect(v, result, select);
#else
  Vector r;
  r.f32[0] = sqrtf(v.f32[0]);
  r.f32[1] = sqrtf(v.f32[1]);
  r.f32[2] = sqrtf(v.f32[2]);
  r.f32[3] = sqrtf(v.f32[3]);
  return r;
#endif
}
LLX_INLINE Vector VectorSqrt(FVector v) {
#if USE_ARM_NEON
  // 3 iterations of Newton-Raphson refinment of sqrt

  // 3次牛顿迭代求 1 / sqrt(k)
  float32x4_t s0 = vrsqrteq_f32(v);
  float32x4_t p0 = vmulq_f32(v, s0);
  float32x4_t r0 = vrsqrtsq_f32(p0, s0);
  float32x4_t s1 = vmulq_f32(s0, r0);
  float32x4_t p1 = vmulq_f32(v, s1);
  float32x4_t r1 = vrsqrtsq_f32(p1, s1);
  float32x4_t s2 = vmulq_f32(s1, r1);
  float32x4_t p2 = vmulq_f32(v, s2);
  float32x4_t r2 = vrsqrtsq_f32(p2, s2);
  float32x4_t s3 = vmulq_f32(s2, r2);

  Vector is_infinity = VectorEqualInt(v, kInfinity.v);
  Vector is_zero = VectorEqual(v, vdupq_n_f32(0));
  // sqrt(k) = k * [1 / sqrt(k)]
  Vector result = vmulq_f32(v, s3);
  Vector select = VectorEqualInt(is_infinity, is_zero);
  return VectorSelect(v, result, select);
#else
  Vector r;
  r.f32[0] = sqrtf(v.f32[0]);
  r.f32[1] = sqrtf(v.f32[1]);
  r.f32[2] = sqrtf(v.f32[2]);
  r.f32[3] = sqrtf(v.f32[3]);
  return r;
#endif
}
LLX_INLINE Vector VectorReciprocalSqrtEst(FVector v) {
#if USE_ARM_NEON
  return vrsqrteq_f32(v);
#else
  Vector r;
  r.f32[0] = 1.f / sqrtf(v.f32[0]);
  r.f32[1] = 1.f / sqrtf(v.f32[1]);
  r.f32[2] = 1.f / sqrtf(v.f32[2]);
  r.f32[3] = 1.f / sqrtf(v.f32[3]);
  return r;
#endif
}

// 牛顿迭代法 x(n+1) = x(n) - f(x(n)) / f'(x(n))
// 求 1 / sqrt(k)
// 假定值为x, 则有 x = 1 / sqrt(k)
// 即 x^(-2) - k = 0
// 有 f(x) = x^(-2) - k
// f'(x) = -2 x^(-3)
// x(n+1) = [(3 - k * x(n) * x(n)) / 2] * x(n)
LLX_INLINE Vector VectorReciprocalSqrt(FVector v) {
#if USE_ARM_NEON
  // 2 iterations of Newton-Raphson refinement of reciprocal
  // 牛顿迭代法: 迭代2次

  // vrsqrteq_f32 估算迭代初值(x(0))
  float32x4_t s0 = vrsqrteq_f32(v);

  // 计算 k * x(0)
  float32x4_t p0 = vmulq_f32(v, s0);

  // 函数vrsqrtsq_f32(a, b)执行计算 (3 - a * b) / 2
  float32x4_t r0 = vrsqrtsq_f32(p0, s0);

  // s1为迭代一次的结果
  float32x4_t s1 = vmulq_f32(s0, r0);
  float32x4_t p1 = vmulq_f32(v, s1);
  // 进行第二次迭代
  float32x4_t r1 = vrsqrtsq_f32(p1, s1);

  // 返回第二次迭代的结果
  return vmulq_f32(s1, r1);
#else
  Vector r;
  r.f32[0] = 1.f / sqrtf(v.f32[0]);
  r.f32[1] = 1.f / sqrtf(v.f32[1]);
  r.f32[2] = 1.f / sqrtf(v.f32[2]);
  r.f32[3] = 1.f / sqrtf(v.f32[3]);
  return r;
#endif
}

LLX_INLINE bool Vector3Equal(FVector v1, FVector v2) {
#if USE_ARM_NEON
  uint32x4_t r = vceqq_f32(v1, v2);
  // uint8x8_t vget_high_u8(uint8x16_t)
  // uint8x8_t vget_low_u8(uint8x16_t)
  // uint8x8x2_t vzip_u8(uint8x8_t, uint8x8_t)
  uint8x8x2_t tmp = vzip_u8(vget_low_u8(reinterpret_cast<uint8x16_t>(r)),
                            vget_high_u8(reinterpret_cast<uint8x16_t>(r)));
  // uint16x4x2_t vzip_u16(uint16x4_t uint16x4_t)
  uint16x4x2_t tmp0 = vzip_u16(reinterpret_cast<uint16x4_t>(tmp.val[0]),
                               reinterpret_cast<uint16x4_t>(tmp.val[1]));
  return ((vget_lane_u32(tmp0.val[1], 1) & 0xFFFFFFU) == 0xFFFFFFU);
#else
  return (((v1.f32[0] == v2.f32[0]) && (v1.f32[1] == v2.f32[1]) &&
           (v1.f32[2] == v2.f32[2])) != 0);
#endif
}
LLX_INLINE bool Vector3EqualInt(FVector v1, FVector v2) {
#if USE_ARM_NEON
  uint32x4_t r = vceqq_u32(v1, v2);
  uint8x8x2_t tmp = vzip_u8(vget_low_u8(r), vget_high_u8(r));
  uint16x4x2_t tmp0 = vzip_u16(tmp.val[0], tmp.val[1]);
  return ((vget_lane_u32(tmp0.val[1], 1) & 0xFFFFFFU) == 0xFFFFFFU);
#else
  return (((v1.u32[0] == v2.u32[0]) && (v1.u32[1] == v2.u32[1]) &&
           (v1.u32[2] == v2.u32[2])) != 0);
#endif
}
LLX_INLINE bool Vector3NearEqual(FVector v1, FVector v2, FVector epsilon) {
#if USE_ARM_NEON
  float32x4_t d = vsubq_f32(v1, v2);
  uint32x4_t r = vcleq_f32(d, epsilon);
  uint8x8x2_t tmp = vzip_u8(vget_low_u8(r), vget_high_u8(r));
  uint16x4x2_t tmp0 = vzip_u16(tmp.val[0], tmp.val[1]);
  return ((vget_lane_u32(tmp0.val[1], 1) & 0xFFFFFFU) == 0xFFFFFFU);
#else
  float dx = fabsf(v1.f32[0] - v2.f32[0]);
  float dy = fabsf(v1.f32[1] - v2.f32[1]);
  float dz = fabsf(v1.f32[2] - v2.f32[2]);
  return ((dx <= epsilon.f32[0]) && (dy <= epsilon.f32[1]) &&
          (dz <= epsilon.f32[2]));
#endif
}
LLX_INLINE bool Vector3NotEqual(FVector v1, FVector v2) {
#if USE_ARM_NEON
  uint32x4_t r = vceqq_f32(v1, v2);
  uint8x8x2_t tmp = vzip_u8(vget_low_u8(r), vget_high_u8(r));
  uint16x4x2_t tmp0 = vzip_u16(tmp.val[0], tmp.val[1]);
  return ((vget_lane_u32(tmp0.val[1], 1) & 0xFFFFFFU) != 0xFFFFFFU);
#else
  return (((v1.f32[0] != v2.f32[0]) || (v1.f32[1] != v2.f32[1]) ||
           (v1.f32[2] != v2.f32[2])) != 0);
#endif
}
LLX_INLINE bool Vector3IsNaN(FVector v) {
#if USE_ARM_NEON
  // Test against itself. NaN is always not equal
  uint32x4_t tmp_nan = vceqq_f32(v, v);
  uint8x8x2_t tmp = vzip_u8(vget_low_u8(tmp_nan), vget_high_u8(tmp_nan));
  uint16x4x2_t tmp0 = vzip_u16(tmp.val[0], tmp.val[1]);
  // If x or y or z are NaN, the mask is zero
  return ((vget_lane_u32(tmp0.val[1], 1) & 0xFFFFFFU) != 0xFFFFFFU);
#else
  return (LLX_IS_NAN(v.f32[0]) || LLX_IS_NAN(v.f32[1]) || LLX_IS_NAN(v.f32[2]));
#endif
}
LLX_INLINE bool Vector3IsInfinite(FVector v) {
#if USE_ARM_NEON
  // Mask off the sign bit
  uint32x4_t tmp_inf = vandq_u32(v, kAbsMask);
  // Compare to infinity
  tmp_inf = vceqq_f32(tmp_inf, kInfinity);
  // If any are infinity, the signs are true.
  uint8x8x2_t tmp = vzip_u8(vget_low_u8(tmp_inf), vget_high_u8(tmp_inf));
  uint16x4x2_t tmp0 = vzip_u16(tmp.val[0], tmp.val[1]);
  return ((vget_lane_u32(tmp0.val[1], 1) & 0xFFFFFFU) != 0);
#else
  return (LLX_IS_INF(v.f32[0]) || LLX_IS_INF(v.f32[1]) || LLX_IS_INF(v.f32[2]));
#endif
}

LLX_INLINE Vector Vector3Dot(FVector v1, FVector v2) {
#if USE_ARM_NEON
  float32x4_t tmp = vmulq_f32(v1, v2);
  float32x2_t vl = vget_low_f32(tmp);
  float32x2_t vh = vget_high_f32(tmp);
  vl = vpadd_f32(vl, vl);
  vh = vdup_lane_f32(vh, 0);
  vl = vadd_f32(vl, vh);
  return vcombine_f32(vl, vl);
#else
  float dp = v1.f32[0] * v2.f32[0] +
             v1.f32[1] * v2.f32[1] +
             v1.f32[2] * v2.f32[2];
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = dp;
  return r;
#endif
}
LLX_INLINE Vector Vector3Cross(FVector v1, FVector v2) {
  // [ V1.y*V2.z - V1.z*V2.y, V1.z*V2.x - V1.x*V2.z, V1.x*V2.y - V1.y*V2.x ]
#if USE_ARM_NEON
  float32x2_t v1xy = vget_low_f32(v1);
  float32x2_t v2xy = vget_low_f32(v2);

  float32x2_t v1yx = vrev64_f32(v1xy);
  float32x2_t v2yx = vrev64_f32(v2xy);

  float32x2_t v1zz = vdup_lane_f32(vget_high_f32(v1), 0);
  float32x2_t v2zz = vdup_lane_f32(vget_high_f32(v2), 0);

  Vector r = vmulq_f32(vcombine_f32(v1yx, v1xy), vcombine_f32(v2zz, v2yx));
  r = vmlsq_f32(r, vcombine_f32(v1zz, v1yx), vcombine_f32(v2yx, v2xy));
  r = veorq_u32(r, kFlipY);
  return vandq_u32(r, kMask3);
#else
  Vector r = {
      (v1.f32[1] * v2.f32[2]) - (v1.f32[2] * v2.f32[1]),
      (v1.f32[2] * v2.f32[0]) - (v1.f32[0] * v2.f32[2]),
      (v1.f32[0] * v2.f32[1]) - (v1.f32[1] * v2.f32[0]),
      0.f
  };
  return r;
#endif
}
LLX_INLINE Vector Vector3LengthSq(FVector v) {
#if USE_ARM_NEON

#else

#endif
  return Vector3Dot(v, v);
}
LLX_INLINE Vector Vector3ReciprocalLengthEst(FVector v) {
#if USE_ARM_NEON
  // Dot3
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vdup_lane_f32(v2, 0);
  v1 = vadd_f32(v1, v2);
  // Reciprocal sqrt (estimate)
  v2 = vrsqrte_f32(v1);
  return vcombine_f32(v2, v2);
#else
  Vector r = Vector3LengthSq(v);
  r = VectorReciprocalSqrtEst(r);
  return r;
#endif
}
LLX_INLINE Vector Vector3ReciprocalLength(FVector v) {
#if USE_ARM_NEON
  // Dot3
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vdup_lane_f32(v2, 0);
  v1 = vadd_f32(v1, v2);
  // Reciprocal sqrt
  float32x2_t S0 = vrsqrte_f32(v1);
  float32x2_t P0 = vmul_f32(v1, S0);
  float32x2_t R0 = vrsqrts_f32(P0, S0);
  float32x2_t S1 = vmul_f32(S0, R0);
  float32x2_t P1 = vmul_f32(v1, S1);
  float32x2_t R1 = vrsqrts_f32(P1, S1);
  float32x2_t Result = vmul_f32(S1, R1);
  return vcombine_f32(Result, Result);
#else
  Vector r = Vector3LengthSq(v);
  r = VectorReciprocalSqrt(r);
  return r;
#endif
}
LLX_INLINE Vector Vector3LengthEst(FVector v) {
#if USE_ARM_NEON
  // Dot3
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vdup_lane_f32(v2, 0);
  v1 = vadd_f32(v1, v2);
  const float32x2_t zero = vdup_n_f32(0);
  uint32x2_t VEqualsZero = vceq_f32(v1, zero);
  // Sqrt (estimate)
  float32x2_t Result = vrsqrte_f32(v1);
  Result = vmul_f32(v1, Result);
  Result = vbsl_f32(VEqualsZero, zero, Result);
  return vcombine_f32(Result, Result);
#else
  Vector r = Vector3LengthSq(v);
  r = VectorSqrtEst(r);
  return r;
#endif
}
LLX_INLINE Vector Vector3Length(FVector v) {
#if USE_ARM_NEON
  // Dot3
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vdup_lane_f32(v2, 0);
  v1 = vadd_f32(v1, v2);
  const float32x2_t zero = vdup_n_f32(0);
  uint32x2_t VEqualsZero = vceq_f32(v1, zero);
  // Sqrt
  float32x2_t S0 = vrsqrte_f32(v1);
  float32x2_t P0 = vmul_f32(v1, S0);
  float32x2_t R0 = vrsqrts_f32(P0, S0);
  float32x2_t S1 = vmul_f32(S0, R0);
  float32x2_t P1 = vmul_f32(v1, S1);
  float32x2_t R1 = vrsqrts_f32(P1, S1);
  float32x2_t Result = vmul_f32(S1, R1);
  Result = vmul_f32(v1, Result);
  Result = vbsl_f32(VEqualsZero, zero, Result);
  return vcombine_f32(Result, Result);
#else
  Vector r = Vector3LengthSq(v);
  r = VectorSqrt(r);
  return r;
#endif
}
LLX_INLINE Vector Vector3NormalizeEst(FVector v) {
#if USE_ARM_NEON
  // Dot3
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vdup_lane_f32(v2, 0);
  v1 = vadd_f32(v1, v2);
  // Reciprocal sqrt (estimate)
  v2 = vrsqrte_f32(v1);
  // Normalize
  return vmulq_f32(v, vcombine_f32(v2,v2));
#else
  Vector r = Vector3ReciprocalLength(v);
  r = VectorMultiply(v, r);
  return r;
#endif
}
LLX_INLINE Vector Vector3Normalize(FVector v) {
#if USE_ARM_NEON
  // Dot3
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vdup_lane_f32(v2, 0);
  v1 = vadd_f32(v1, v2);
  uint32x2_t VEqualsZero = vceq_f32(v1, vdup_n_f32(0));
  uint32x2_t VEqualsInf = vceq_f32(v1, vget_low_f32(kInfinity));
  // Reciprocal sqrt (2 iterations of Newton-Raphson)
  float32x2_t S0 = vrsqrte_f32(v1);
  float32x2_t P0 = vmul_f32(v1, S0);
  float32x2_t R0 = vrsqrts_f32(P0, S0);
  float32x2_t S1 = vmul_f32(S0, R0);
  float32x2_t P1 = vmul_f32(v1, S1);
  float32x2_t R1 = vrsqrts_f32(P1, S1);
  v2 = vmul_f32(S1, R1);
  // Normalize
  Vector vResult = vmulq_f32(v, vcombine_f32(v2, v2));
  vResult = vbslq_f32(
      vcombine_f32(VEqualsZero, VEqualsZero), vdupq_n_f32(0), vResult);
  return vbslq_f32(vcombine_f32(VEqualsInf, VEqualsInf), kQNaN, vResult);
#else
  Vector r = Vector3Length(v);
  float len = r.f32[0];
  if (len > 0) {
    len = 1.f / len;
  }

  r.f32[0] = v.f32[0] * len;
  r.f32[1] = v.f32[1] * len;
  r.f32[2] = v.f32[2] * len;
  r.f32[3] = v.f32[3] * len;
  return r;
#endif
}

LLX_INLINE Vector Vector3Reflect(FVector incident, FVector normal) {
  // Result = Incident - (2 * dot(Incident, Normal)) * Normal
  Vector r = Vector3Dot(incident, normal);
  r = VectorAdd(r, r);
  r = VectorNegativeMultiplySubtract(r, normal, incident);
  return r;
}

LLX_INLINE bool Vector4Equal(FVector v1, FVector v2) {
#if USE_ARM_NEON
  uint32x4_t vResult = vceqq_f32(v1, v2);
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vResult), vget_high_u8(vResult));
  uint16x4x2_t tmp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  return (vget_lane_u32(tmp0.val[1], 1) == 0xFFFFFFFFU);
#else
  return ((v1.f32[0] == v2.f32[0]) && (v1.f32[1] == v2.f32[1]) &&
          (v1.f32[2] == v2.f32[2]) && (v1.f32[3] == v2.f32[3]));
#endif
}
LLX_INLINE bool Vector4EqualInt(FVector v1, FVector v2) {
#if USE_ARM_NEON
  uint32x4_t vResult = vceqq_u32(v1, v2);
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vResult), vget_high_u8(vResult));
  uint16x4x2_t tmp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  return (vget_lane_u32(tmp0.val[1], 1) == 0xFFFFFFFFU);
#else
  return ((v1.u32[0] == v2.u32[0]) && (v1.u32[1] == v2.u32[1]) &&
          (v1.u32[2] == v2.u32[2]) && (v1.u32[3] == v2.u32[3]));
#endif
}
LLX_INLINE bool Vector4NearEqual(FVector v1, FVector v2, FVector epsilon) {
#if USE_ARM_NEON
  float32x4_t vDelta = vsubq_f32(v1, v2);
  uint32x4_t vResult = vcleq_f32(vDelta, epsilon);
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vResult), vget_high_u8(vResult));
  uint16x4x2_t tmp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  return (vget_lane_u32(tmp0.val[1], 1) == 0xFFFFFFFFU);
#else
  float dx = fabsf(v1.f32[0] - v2.f32[0]);
  float dy = fabsf(v1.f32[1] - v2.f32[1]);
  float dz = fabsf(v1.f32[2] - v2.f32[2]);
  float dw = fabsf(v1.f32[3] - v2.f32[3]);
  return ((dx <= epsilon.f32[0]) && (dy <= epsilon.f32[1]) &&
          (dz <= epsilon.f32[2]) && (dw <= epsilon.f32[3]));
#endif
}
LLX_INLINE bool Vector4NotEqual(FVector v1, FVector v2) {
#if USE_ARM_NEON
  uint32x4_t vResult = vceqq_f32(v1, v2);
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vResult), vget_high_u8(vResult));
  uint16x4x2_t tmp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  return (vget_lane_u32(tmp0.val[1], 1) != 0xFFFFFFFFU);
#else
  return ((v1.f32[0] != v2.f32[0]) || (v1.f32[1] != v2.f32[1]) ||
          (v1.f32[2] != v2.f32[2]) || (v1.f32[3] != v2.f32[3]));
#endif
}
LLX_INLINE bool Vector4IsNaN(FVector v) {
#if USE_ARM_NEON
  // Test against itself. NaN is always not equal
  uint32x4_t vTempNan = vceqq_f32(v, v);
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vTempNan), vget_high_u8(vTempNan));
  uint16x4x2_t tmp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  // If any are NaN, the mask is zero
  return (vget_lane_u32(tmp0.val[1], 1) != 0xFFFFFFFFU);
#else
  return (LLX_IS_NAN(v.f32[0]) || LLX_IS_NAN(v.f32[1]) ||
          LLX_IS_NAN(v.f32[2]) || LLX_IS_NAN(v.f32[3]));
#endif
}
LLX_INLINE bool Vector4IsInfinite(FVector v) {
#if USE_ARM_NEON
  // Mask off the sign bit
  uint32x4_t vTempInf = vandq_u32(v, kAbsMask);
  // Compare to infinity
  vTempInf = vceqq_f32(vTempInf, kInfinity);
  // If any are infinity, the signs are true.
  uint8x8x2_t vTemp = vzip_u8(vget_low_u8(vTempInf), vget_high_u8(vTempInf));
  uint16x4x2_t tmp0 = vzip_u16(vTemp.val[0], vTemp.val[1]);
  return (vget_lane_u32(tmp0.val[1], 1) != 0);
#else
  return (LLX_IS_INF(v.f32[0]) || LLX_IS_INF(v.f32[1]) ||
          LLX_IS_INF(v.f32[2]) || LLX_IS_INF(v.f32[3]));
#endif
}
LLX_INLINE Vector Vector4Dot(FVector v1, FVector v2) {
#if USE_ARM_NEON
  float32x4_t vTemp = vmulq_f32(v1, v2);
  float32x2_t vl = vget_low_f32(vTemp);
  float32x2_t vh = vget_high_f32(vTemp);
  vl = vpadd_f32(vl, vl);
  vh = vpadd_f32(vh, vh);
  vl = vadd_f32(vl, vh);
  return vcombine_f32(vl, vl);
#else
  float dp = v1.f32[0] * v2.f32[0] + v1.f32[1] * v2.f32[1] +
             v1.f32[2] * v2.f32[2] + v1.f32[3] * v2.f32[3];
  Vector r;
  r.f32[0] = r.f32[1] = r.f32[2] = r.f32[3] = dp;
  return r;
#endif
}
LLX_INLINE Vector Vector4Cross(FVector v1, FVector v2, FVector v3) {
  // [ ((v2.z*v3.w-v2.w*v3.z)*v1.y)-((v2.y*v3.w-v2.w*v3.y)*v1.z)+((v2.y*v3.z-v2.z*v3.y)*v1.w),
  //   ((v2.w*v3.z-v2.z*v3.w)*v1.x)-((v2.w*v3.x-v2.x*v3.w)*v1.z)+((v2.z*v3.x-v2.x*v3.z)*v1.w),
  //   ((v2.y*v3.w-v2.w*v3.y)*v1.x)-((v2.x*v3.w-v2.w*v3.x)*v1.y)+((v2.x*v3.y-v2.y*v3.x)*v1.w),
  //   ((v2.z*v3.y-v2.y*v3.z)*v1.x)-((v2.z*v3.x-v2.x*v3.z)*v1.y)+((v2.y*v3.x-v2.x*v3.y)*v1.z) ]
#if USE_ARM_NEON
  const float32x2_t select = vget_low_f32(kMaskX);

  // Term1: V2zwyz * V3wzwy
  const float32x2_t v2xy = vget_low_f32(v2);
  const float32x2_t v2zw = vget_high_f32(v2);
  const float32x2_t v2yx = vrev64_f32(v2xy);
  const float32x2_t v2wz = vrev64_f32(v2zw);
  const float32x2_t v2yz = vbsl_f32(select, v2yx, v2wz);

  const float32x2_t v3zw = vget_high_f32(v3);
  const float32x2_t v3wz = vrev64_f32(v3zw);
  const float32x2_t v3xy = vget_low_f32(v3);
  const float32x2_t v3wy = vbsl_f32(select, v3wz, v3xy);

  float32x4_t vTemp1 = vcombine_f32(v2zw, v2yz);
  float32x4_t vTemp2 = vcombine_f32(v3wz, v3wy);
  Vector vResult = vmulq_f32(vTemp1, vTemp2);

  // - V2wzwy * V3zwyz
  const float32x2_t v2wy = vbsl_f32(select, v2wz, v2xy);

  const float32x2_t v3yx = vrev64_f32(v3xy);
  const float32x2_t v3yz = vbsl_f32(select, v3yx, v3wz);

  vTemp1 = vcombine_f32(v2wz, v2wy);
  vTemp2 = vcombine_f32(v3zw, v3yz);
  vResult = vmlsq_f32(vResult, vTemp1, vTemp2);

  // term1 * V1yxxx
  const float32x2_t v1xy = vget_low_f32(v1);
  const float32x2_t v1yx = vrev64_f32(v1xy);

  vTemp1 = vcombine_f32(v1yx, vdup_lane_f32(v1yx, 1));
  vResult = vmulq_f32(vResult, vTemp1);

  // Term2: V2ywxz * V3wxwx
  const float32x2_t v2yw = vrev64_f32(v2wy);
  const float32x2_t v2xz = vbsl_f32(select, v2xy, v2wz);

  const float32x2_t v3wx = vbsl_f32(select, v3wz, v3yx);

  vTemp1 = vcombine_f32(v2yw, v2xz);
  vTemp2 = vcombine_f32(v3wx, v3wx);
  float32x4_t vTerm = vmulq_f32(vTemp1, vTemp2);

  // - V2wxwx * V3ywxz
  const float32x2_t v2wx = vbsl_f32(select, v2wz, v2yx);

  const float32x2_t v3yw = vrev64_f32(v3wy);
  const float32x2_t v3xz = vbsl_f32(select, v3xy, v3wz);

  vTemp1 = vcombine_f32(v2wx, v2wx);
  vTemp2 = vcombine_f32(v3yw, v3xz);
  vTerm = vmlsq_f32(vTerm, vTemp1, vTemp2);

  // vResult - term2 * V1zzyy
  const float32x2_t v1zw = vget_high_f32(v1);

  vTemp1 = vcombine_f32(vdup_lane_f32(v1zw, 0), vdup_lane_f32(v1yx, 0));
  vResult = vmlsq_f32(vResult, vTerm, vTemp1);

  // Term3: V2yzxy * V3zxyx
  const float32x2_t v3zx = vrev64_f32(v3xz);

  vTemp1 = vcombine_f32(v2yz, v2xy);
  vTemp2 = vcombine_f32(v3zx, v3yx);
  vTerm = vmulq_f32(vTemp1, vTemp2);

  // - V2zxyx * V3yzxy
  const float32x2_t v2zx = vrev64_f32(v2xz);

  vTemp1 = vcombine_f32(v2zx, v2yx);
  vTemp2 = vcombine_f32(v3yz, v3xy);
  vTerm = vmlsq_f32(vTerm, vTemp1, vTemp2);

  // vResult + term3 * V1wwwz
  const float32x2_t v1wz = vrev64_f32(v1zw);

  vTemp1 = vcombine_f32(vdup_lane_f32(v1wz, 0), v1wz);
  return vmlaq_f32(vResult, vTerm, vTemp1);
#else
  Vector r;
  r.f32[0] =
      (((v2.f32[2] * v3.f32[3]) - (v2.f32[3] * v3.f32[2])) * v1.f32[1]) -
      (((v2.f32[1] * v3.f32[3]) - (v2.f32[3] * v3.f32[1])) * v1.f32[2]) +
      (((v2.f32[1] * v3.f32[2]) - (v2.f32[2] * v3.f32[1])) * v1.f32[3]);
  r.f32[1] =
      (((v2.f32[3] * v3.f32[2]) - (v2.f32[2] * v3.f32[3])) * v1.f32[0]) -
      (((v2.f32[3] * v3.f32[0]) - (v2.f32[0] * v3.f32[3])) * v1.f32[2]) +
      (((v2.f32[2] * v3.f32[0]) - (v2.f32[0] * v3.f32[2])) * v1.f32[3]);
  r.f32[2] =
      (((v2.f32[1] * v3.f32[3]) - (v2.f32[3] * v3.f32[1])) * v1.f32[0]) -
      (((v2.f32[0] * v3.f32[3]) - (v2.f32[3] * v3.f32[0])) * v1.f32[1]) +
      (((v2.f32[0] * v3.f32[1]) - (v2.f32[1] * v3.f32[0])) * v1.f32[3]);
  r.f32[3] =
      (((v2.f32[2] * v3.f32[1]) - (v2.f32[1] * v3.f32[2])) * v1.f32[0]) -
      (((v2.f32[2] * v3.f32[0]) - (v2.f32[0] * v3.f32[2])) * v1.f32[1]) +
      (((v2.f32[1] * v3.f32[0]) - (v2.f32[0] * v3.f32[1])) * v1.f32[2]);
  return r;
#endif
}
LLX_INLINE Vector Vector4LengthSq(FVector v) {
#if USE_ARM_NEON

#else

#endif
  return Vector4Dot(v, v);
}
LLX_INLINE Vector Vector4ReciprocalLengthEst(FVector v) {
#if USE_ARM_NEON
  // Dot4
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vpadd_f32(v2, v2);
  v1 = vadd_f32(v1, v2);
  // Reciprocal sqrt (estimate)
  v2 = vrsqrte_f32(v1);
  return vcombine_f32(v2, v2);
#else
  Vector r = Vector4LengthSq(v);
  r = VectorReciprocalSqrtEst(r);
  return r;
#endif
}
LLX_INLINE Vector Vector4ReciprocalLength(FVector v) {
#if USE_ARM_NEON
  // Dot4
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vpadd_f32(v2, v2);
  v1 = vadd_f32(v1, v2);
  // Reciprocal sqrt
  float32x2_t S0 = vrsqrte_f32(v1);
  float32x2_t P0 = vmul_f32(v1, S0);
  float32x2_t R0 = vrsqrts_f32(P0, S0);
  float32x2_t S1 = vmul_f32(S0, R0);
  float32x2_t P1 = vmul_f32(v1, S1);
  float32x2_t R1 = vrsqrts_f32(P1, S1);
  float32x2_t Result = vmul_f32(S1, R1);
  return vcombine_f32(Result, Result);
#else
  Vector r = Vector4LengthSq(v);
  r = VectorReciprocalSqrt(r);
  return r;
#endif
}
LLX_INLINE Vector Vector4LengthEst(FVector v) {
#if USE_ARM_NEON
  // Dot4
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vpadd_f32(v2, v2);
  v1 = vadd_f32(v1, v2);
  const float32x2_t zero = vdup_n_f32(0);
  uint32x2_t VEqualsZero = vceq_f32(v1, zero);
  // Sqrt (estimate)
  float32x2_t Result = vrsqrte_f32(v1);
  Result = vmul_f32(v1, Result);
  Result = vbsl_f32(VEqualsZero, zero, Result);
  return vcombine_f32(Result, Result);
#else
  Vector r = Vector4LengthSq(v);
  r = VectorSqrtEst(r);
  return r;
#endif
}
LLX_INLINE Vector Vector4Length(FVector v) {
#if USE_ARM_NEON
  // Dot4
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vpadd_f32(v2, v2);
  v1 = vadd_f32(v1, v2);
  const float32x2_t zero = vdup_n_f32(0);
  uint32x2_t VEqualsZero = vceq_f32(v1, zero);
  // Sqrt
  float32x2_t S0 = vrsqrte_f32(v1);
  float32x2_t P0 = vmul_f32(v1, S0);
  float32x2_t R0 = vrsqrts_f32(P0, S0);
  float32x2_t S1 = vmul_f32(S0, R0);
  float32x2_t P1 = vmul_f32(v1, S1);
  float32x2_t R1 = vrsqrts_f32(P1, S1);
  float32x2_t Result = vmul_f32(S1, R1);
  Result = vmul_f32(v1, Result);
  Result = vbsl_f32(VEqualsZero, zero, Result);
  return vcombine_f32(Result, Result);
#else
  Vector r = Vector4LengthSq(v);
  r = VectorSqrt(r);
  return r;
#endif
}
LLX_INLINE Vector Vector4NormalizeEst(FVector v) {
#if USE_ARM_NEON
  // Dot4
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vpadd_f32(v2, v2);
  v1 = vadd_f32(v1, v2);
  // Reciprocal sqrt (estimate)
  v2 = vrsqrte_f32(v1);
  // Normalize
  return vmulq_f32(v, vcombine_f32(v2, v2));
#else
  Vector r = Vector4ReciprocalLength(v);
  r = VectorMultiply(v, r);
  return r;
#endif
}
LLX_INLINE Vector Vector4Normalize(FVector v) {
#if USE_ARM_NEON
  // Dot4
  float32x4_t vTemp = vmulq_f32(v, v);
  float32x2_t v1 = vget_low_f32(vTemp);
  float32x2_t v2 = vget_high_f32(vTemp);
  v1 = vpadd_f32(v1, v1);
  v2 = vpadd_f32(v2, v2);
  v1 = vadd_f32(v1, v2);
  uint32x2_t VEqualsZero = vceq_f32(v1, vdup_n_f32(0));
  uint32x2_t VEqualsInf = vceq_f32(v1, vget_low_f32(kInfinity));
  // Reciprocal sqrt (2 iterations of Newton-Raphson)
  float32x2_t S0 = vrsqrte_f32(v1);
  float32x2_t P0 = vmul_f32(v1, S0);
  float32x2_t R0 = vrsqrts_f32(P0, S0);
  float32x2_t S1 = vmul_f32(S0, R0);
  float32x2_t P1 = vmul_f32(v1, S1);
  float32x2_t R1 = vrsqrts_f32(P1, S1);
  v2 = vmul_f32(S1, R1);
  // Normalize
  Vector vResult = vmulq_f32(v, vcombine_f32(v2, v2));
  vResult = vbslq_f32(vcombine_f32(VEqualsZero, VEqualsZero),
                      vdupq_n_f32(0), vResult);
  return vbslq_f32(vcombine_f32(VEqualsInf, VEqualsInf), kQNaN, vResult);
#else
  Vector r = Vector4Length(v);
  float len = r.f32[0];
  if (len > 0) {
    len = 1.f / len;
  }
  r.f32[0] = v.f32[0] * len;
  r.f32[1] = v.f32[1] * len;
  r.f32[2] = v.f32[2] * len;
  r.f32[3] = v.f32[3] * len;
  return r;
#endif
}
LLX_INLINE Vector Vector4Reflect(FVector incident, FVector normal) {
  // Result = Incident - (2 * dot(Incident, Normal)) * Normal

  Vector r = Vector4Dot(incident, normal);
  r = VectorAdd(r, r);
  r = VectorNegativeMultiplySubtract(r, normal, incident);
  return r;
}

/*LLX_INLINE Vector operator+ (FVector v) {
  return v;
}
LLX_INLINE Vector operator- (FVector v) {
  return VectorNegate(v);
}

LLX_INLINE Vector& operator+= (Vector& v1, FVector v2) {
  v1 = VectorAdd(v1, v2);
  return v1;
}
LLX_INLINE Vector& operator-= (Vector& v1, FVector v2) {
  v1 = VectorSubtract(v1, v2);
  return v1;
}
LLX_INLINE Vector& operator*= (Vector& v1, FVector v2) {
  v1 = VectorMultiply(v1, v2);
  return v1;
}
LLX_INLINE Vector& operator/= (Vector& v1, FVector v2) {
  v1 = VectorDivide(v1, v2);
  return v1;
}
LLX_INLINE Vector& operator*= (Vector& v, float s) {
  v = VectorScale(v, s);
  return v;
}
LLX_INLINE Vector& operator/= (Vector& v, float s) {
  Vector vs = VectorReplicate(s);
  v = VectorDivide(v, vs);
  return v;
}

LLX_INLINE Vector operator+ (FVector v1, FVector v2) {
  return VectorAdd(v1, v2);
}
LLX_INLINE Vector operator- (FVector v1, FVector v2) {
  return VectorSubtract(v1, v2);
}
LLX_INLINE Vector operator* (FVector v1, FVector v2) {
  return VectorMultiply(v1, v2);
}
LLX_INLINE Vector operator/ (FVector v1, FVector v2) {
  return VectorDivide(v1, v2);
}
LLX_INLINE Vector operator* (FVector v, float s) {
  return VectorScale(v, s);
}
LLX_INLINE Vector operator* (float s, FVector v) {
  return VectorScale(v, s);
}
LLX_INLINE Vector operator/ (FVector v, float s) {
  Vector vs = VectorReplicate(s);
  return VectorDivide(v, vs);
}*/
}
}
#endif