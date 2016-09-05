//
// Created by Marco.LIU on 16/8/16.
// Copyright (c) 2016 LLX. All rights reserved.
//

#ifndef _LLX_UTIL_MATRIX_VECTORS_H_
#define _LLX_UTIL_MATRIX_VECTORS_H_

#include <array>
#include <cstring>
#include <math.h>

#include "matrix_config.h"

namespace LLX {
namespace util {
#if USE_ARM_NEON
typedef float32x4_t Vector;

// param 1 - 3
typedef const Vector FVector;
// param 4
typedef const Vector GVector;

// param 5 - 6
#ifdef __aarch64__
typedef const Vector HVector;
#else
typedef const Vector& HVector;
#endif
#else
typedef struct __Vector {
  union {
    std::array<float, 4>    f32;
    std::array<uint32_t, 4> u32;
  };
} Vector;

// param 1 - 3
typedef const Vector& FVector;
// param 4
typedef const Vector& GVector;
// param 5 - 6
typedef const Vector& HVector;
#endif

// param 7+
typedef const Vector& CVector;

struct __attribute__((aligned(16))) VectorF32 {
  inline operator Vector() const { return v; }
  inline const float* data() const { return f; }

  union {
    float f[4];
    Vector v;
  };
};

struct __attribute__((aligned(16))) VectorI32 {
  inline operator Vector() const { return v; }

  union {
    int32_t i[4];
    Vector v;
  };
};

struct __attribute__((aligned(16))) VectorU32 {
  inline operator Vector() const { return v; }

  union {
    uint32_t i[4];
    Vector v;
  };
};



struct Vector2f {
  Vector2f() = default;
  Vector2f(float a, float b) : x(a), y(b) {}
  float x, y;
};

struct __attribute__((aligned(16))) Vector2fA : public Vector2f {
  Vector2fA() = default;
  Vector2fA(float a, float b) : Vector2f(a, b) {}
};

struct Vector2i {
  Vector2i() = default;
  Vector2i(int32_t a, int32_t b) : x(a), y(b) {}

  int32_t x, y;
};

struct Vector2u {
  Vector2u() = default;
  Vector2u(uint32_t a, uint32_t b) : x(a), y(b) {}

  uint32_t x, y;
};

struct Vector3f : public Vector2f {
  Vector3f() = default;
  Vector3f(float a, float b, float c) : Vector2f(a, b), z(c) {}

  float z;
};

struct __attribute__((aligned(16))) Vector3fA : public Vector3f {
  Vector3fA() = default;
  Vector3fA(float a, float b, float c) : Vector3f(a, b, c) {}
};

struct Vector3i : public Vector2i {
  Vector3i() = default;
  Vector3i(int32_t a, int32_t b, int32_t c) : Vector2i(a, b), z(c) {}

  int32_t z;
};

struct Vector3u : public Vector2u {
  Vector3u() = default;
  Vector3u(uint32_t a, uint32_t b, uint32_t c) : Vector2u(a, b), z(c) {}

  uint32_t z;
};

struct Vector4f : public Vector3f {
  Vector4f() = default;
  Vector4f(float a, float b, float c, float d) : Vector3f(a, b, c), w(d) {}
  float w;
};

struct __attribute__((aligned(16))) Vector4fA : public Vector4f {
  Vector4fA() = default;
  Vector4fA(float a, float b, float c, float d) : Vector4f(a, b, c, d) {}
};

struct Vector4i : public Vector3i {
  Vector4i() = default;
  Vector4i(int32_t a, int32_t b, int32_t c, int32_t d)
      : Vector3i(a, b, c), w(d) {}

  int32_t w;
};

struct Vector4u : public Vector3u {
  Vector4u() = default;
  Vector4u(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
      : Vector3u(a, b, c), w(d) {}

  uint32_t w;
};

/*Vector operator+ (FVector v);
Vector operator- (FVector v);

Vector& operator+= (Vector& v1, FVector v2);
Vector& operator-= (Vector& v1, FVector v2);
Vector& operator*= (Vector& v1, FVector v2);
Vector& operator/= (Vector& v1, FVector v2);

Vector& operator*= (Vector& v, float s);
Vector& operator/= (Vector& v, float s);

Vector operator+ (FVector v1, FVector v2);
Vector operator- (FVector v1, FVector v2);
Vector operator* (FVector v1, FVector v2);
Vector operator/ (FVector v1, FVector v2);
Vector operator* (FVector v, float s);
Vector operator* (float s, FVector v);
Vector operator/ (FVector v, float s);*/

float ScalarSin(float radian);
float ScalarSinEst(float radian);
float ScalarCos(float radian);
float ScalarCosEst(float radian);
void ScalarSinCos(float* sine, float* cosine, float radian);
void ScalarSinCosEst(float* sine, float* cosine, float radian);
float ScalarASin(float sine);
float ScalarASinEst(float sine);
float ScalarACos(float cosine);
float ScalarACosEst(float cosine);


Vector VectorZero();
Vector VectorSet(float x, float y, float z, float w);
Vector VectorReplicate(float val);
Vector VectorReplicate(const float* val);
Vector VectorReplicate(uint32_t val);
Vector VectorReplicate(const uint32_t* val);
Vector VectorSplatX(FVector v);
Vector VectorSplatY(FVector v);
Vector VectorSplatZ(FVector v);
Vector VectorSplatW(FVector v);
Vector VectorSplatOne();
Vector VectorSplatInfinit();
Vector VectorSplatQNaN();
Vector VectorSplatEpsilon();
Vector VectorSplatSignMask();

//float VectorGetByIndex(FVector v, size_t i);
float VectorGetX(FVector v);
float VectorGetY(FVector v);
float VectorGetZ(FVector v);
float VectorGetW(FVector v);

//void VectorGetByIndex(float* f, FVector v, size_t i);
void VectorGetX(float* f, FVector v);
void VectorGetY(float* f, FVector v);
void VectorGetZ(float* f, FVector v);
void VectorGetW(float* f, FVector v);

//Vector VectorSetByIndex(FVector v, float f, size_t i);
Vector VectorSetX(FVector v, float f);
Vector VectorSetY(FVector v, float f);
Vector VectorSetZ(FVector v, float f);
Vector VectorSetW(FVector v, float f);

//Vector VectorSetByIndex(FVector v, const float* f, size_t i);
Vector VectorSetX(FVector v, const float* f);
Vector VectorSetY(FVector v, const float* f);
Vector VectorSetZ(FVector v, const float* f);
Vector VectorSetW(FVector v, const float* f);

Vector VectorSwizzle(FVector v, uint32_t e0, uint32_t e1, uint32_t e2,
                     uint32_t e3);

template <uint32_t SwizzleX, uint32_t SwizzleY, uint32_t SwizzleZ,
    uint32_t SwizzleW>
inline Vector VectorSwizzle(FVector v) {
  static_assert(SwizzleX <= 3, "SwizzleX template parameter out of range");
  static_assert(SwizzleY <= 3, "SwizzleY template parameter out of range");
  static_assert(SwizzleZ <= 3, "SwizzleZ template parameter out of range");
  static_assert(SwizzleW <= 3, "SwizzleW template parameter out of range");
  return VectorSwizzle(v, SwizzleX, SwizzleY, SwizzleZ, SwizzleW);
};

template<> inline Vector VectorSwizzle<0,1,2,3>(FVector v) { return v; }

#if USE_ARM_NEON
template<>
inline Vector VectorSwizzle<0, 0, 0, 0>(FVector v) {
  return vdupq_lane_f32(vget_low_f32(v), 0);
}

template<>
inline Vector VectorSwizzle<1, 1, 1, 1>(FVector v) {
  return vdupq_lane_f32(vget_low_f32(v), 1);
}

template<>
inline Vector VectorSwizzle<2, 2, 2, 2>(FVector v) {
  return vdupq_lane_f32(vget_high_f32(v), 0);
}

template<>
inline Vector VectorSwizzle<3, 3, 3, 3>(FVector v) {
  return vdupq_lane_f32(vget_high_f32(v), 1);
}

template<>
inline Vector VectorSwizzle<1, 0, 3, 2>(FVector v) {
  return vrev64q_f32(v);
}

template<>
inline Vector VectorSwizzle<0, 1, 0, 1>(FVector v) {
  float32x2_t vt = vget_low_f32(v);
  return vcombine_f32(vt, vt);
}

template<>
inline Vector VectorSwizzle<2, 3, 2, 3>(FVector v) {
  float32x2_t vt = vget_high_f32(v);
  return vcombine_f32(vt, vt);
}

template<>
inline Vector VectorSwizzle<1, 0, 1, 0>(FVector v) {
  float32x2_t vt = vrev64_f32(vget_low_f32(v));
  return vcombine_f32(vt, vt);
}

template<>
inline Vector VectorSwizzle<3, 2, 3, 2>(FVector v) {
  float32x2_t vt = vrev64_f32(vget_high_f32(v));
  return vcombine_f32(vt, vt);
}

template<>
inline Vector VectorSwizzle<0, 1, 3, 2>(FVector v) {
  return vcombine_f32(vget_low_f32(v), vrev64_f32(vget_high_f32(v)));
}

template<>
inline Vector VectorSwizzle<1, 0, 2, 3>(FVector v) {
  return vcombine_f32(vrev64_f32(vget_low_f32(v)), vget_high_f32(v));
}

template<>
inline Vector VectorSwizzle<2, 3, 1, 0>(FVector v) {
  return vcombine_f32(vget_high_f32(v), vrev64_f32(vget_low_f32(v)));
}

template<>
inline Vector VectorSwizzle<3, 2, 0, 1>(FVector v) {
  return vcombine_f32(vrev64_f32(vget_high_f32(v)), vget_low_f32(v));
}

template<>
inline Vector VectorSwizzle<3, 2, 1, 0>(FVector v) {
  return vcombine_f32(vrev64_f32(vget_high_f32(v)),
                      vrev64_f32(vget_low_f32(v)));
}

template<>
inline Vector VectorSwizzle<0, 0, 2, 2>(FVector v) {
  return vtrnq_f32(v, v).val[0];
}

template<>
inline Vector VectorSwizzle<1, 1, 3, 3>(FVector v) {
  return vtrnq_f32(v, v).val[1];
}

template<>
inline Vector VectorSwizzle<0, 0, 1, 1>(FVector v) {
  return vzipq_f32(v, v).val[0];
}

template<>
inline Vector VectorSwizzle<2, 2, 3, 3>(FVector v) {
  return vzipq_f32(v, v).val[1];
}

template<>
inline Vector VectorSwizzle<0, 2, 0, 2>(FVector v) {
  return vuzpq_f32(v, v).val[0];
}

template<>
inline Vector VectorSwizzle<1, 3, 1, 3>(FVector v) {
  return vuzpq_f32(v, v).val[1];
}

template<>
inline Vector VectorSwizzle<1, 2, 3, 0>(FVector v) {
  return vextq_f32(v, v, 1);
}

template<>
inline Vector VectorSwizzle<2, 3, 0, 1>(FVector v) {
  return vextq_f32(v, v, 2);
}

template<>
inline Vector VectorSwizzle<3, 0, 1, 2>(FVector v) {
  return vextq_f32(v, v, 3);
}
#endif


Vector VectorPermute(FVector v1, FVector v2, uint32_t permute_x,
                     uint32_t permute_y, uint32_t permute_z,
                     uint32_t permute_w);

template<uint32_t PermuteX, uint32_t PermuteY, uint32_t PermuteZ,
    uint32_t PermuteW>
inline Vector VectorPermute(FVector v1, FVector v2)  {
  static_assert(PermuteX <= 7, "PermuteX template parameter out of range");
  static_assert(PermuteY <= 7, "PermuteY template parameter out of range");
  static_assert(PermuteZ <= 7, "PermuteZ template parameter out of range");
  static_assert(PermuteW <= 7, "PermuteW template parameter out of range");

  return VectorPermute(v1, v2, PermuteX, PermuteY, PermuteZ, PermuteW);
}

template<> inline Vector VectorPermute<0,1,2,3>(FVector v1, FVector v2) {
  return v1;
}
template<> inline Vector VectorPermute<4,5,6,7>(FVector v1, FVector v2) {
  return v2;
}

#if USE_ARM_NEON
template<>
inline Vector VectorPermute<0, 1, 4, 5>(FVector v1, FVector v2) {
  return vcombine_f32(vget_low_f32(v1), vget_low_f32(v2));
}

template<>
inline Vector VectorPermute<1, 0, 4, 5>(FVector v1, FVector v2) {
  return vcombine_f32(vrev64_f32(vget_low_f32(v1)), vget_low_f32(v2));
}

template<>
inline Vector VectorPermute<0, 1, 5, 4>(FVector v1, FVector v2) {
  return vcombine_f32(vget_low_f32(v1), vrev64_f32(vget_low_f32(v2)));
}

template<>
inline Vector VectorPermute<1, 0, 5, 4>(FVector v1, FVector v2) {
  return vcombine_f32(vrev64_f32(vget_low_f32(v1)),
                      vrev64_f32(vget_low_f32(v2)));
}

template<>
inline Vector VectorPermute<2, 3, 6, 7>(FVector v1, FVector v2) {
  return vcombine_f32(vget_high_f32(v1), vget_high_f32(v2));
}

template<>
inline Vector VectorPermute<3, 2, 6, 7>(FVector v1, FVector v2) {
  return vcombine_f32(vrev64_f32(vget_high_f32(v1)), vget_high_f32(v2));
}

template<>
inline Vector VectorPermute<2, 3, 7, 6>(FVector v1, FVector v2) {
  return vcombine_f32(vget_high_f32(v1), vrev64_f32(vget_high_f32(v2)));
}

template<>
inline Vector VectorPermute<3, 2, 7, 6>(FVector v1, FVector v2) {
  return vcombine_f32(vrev64_f32(vget_high_f32(v1)),
                      vrev64_f32(vget_high_f32(v2)));
}

template<>
inline Vector VectorPermute<0, 1, 6, 7>(FVector v1, FVector v2) {
  return vcombine_f32(vget_low_f32(v1), vget_high_f32(v2));
}

template<>
inline Vector VectorPermute<1, 0, 6, 7>(FVector v1, FVector v2) {
  return vcombine_f32(vrev64_f32(vget_low_f32(v1)), vget_high_f32(v2));
}

template<>
inline Vector VectorPermute<0, 1, 7, 6>(FVector v1, FVector v2) {
  return vcombine_f32(vget_low_f32(v1), vrev64_f32(vget_high_f32(v2)));
}

template<>
inline Vector VectorPermute<1, 0, 7, 6>(FVector v1, FVector v2) {
  return vcombine_f32(vrev64_f32(vget_low_f32(v1)),
                      vrev64_f32(vget_high_f32(v2)));
}

template<>
inline Vector VectorPermute<3, 2, 4, 5>(FVector v1, FVector v2) {
  return vcombine_f32(vrev64_f32(vget_high_f32(v1)), vget_low_f32(v2));
}

template<>
inline Vector VectorPermute<2, 3, 5, 4>(FVector v1, FVector v2) {
  return vcombine_f32(vget_high_f32(v1), vrev64_f32(vget_low_f32(v2)));
}

template<>
inline Vector VectorPermute<3, 2, 5, 4>(FVector v1, FVector v2) {
  return vcombine_f32(vrev64_f32(vget_high_f32(v1)),
                      vrev64_f32(vget_low_f32(v2)));
}

template<>
inline Vector VectorPermute<0, 4, 2, 6>(FVector v1, FVector v2) {
  return vtrnq_f32(v1, v2).val[0];
}

template<>
inline Vector VectorPermute<1, 5, 3, 7>(FVector v1, FVector v2) {
  return vtrnq_f32(v1, v2).val[1];
}

template<>
inline Vector VectorPermute<0, 4, 1, 5>(FVector v1, FVector v2) {
  return vzipq_f32(v1, v2).val[0];
}

template<>
inline Vector VectorPermute<2, 6, 3, 7>(FVector v1, FVector v2) {
  return vzipq_f32(v1, v2).val[1];
}

template<>
inline Vector VectorPermute<0, 2, 4, 6>(FVector v1, FVector v2) {
  return vuzpq_f32(v1, v2).val[0];
}

template<>
inline Vector VectorPermute<1, 3, 5, 7>(FVector v1, FVector v2) {
  return vuzpq_f32(v1, v2).val[1];
}

template<>
inline Vector VectorPermute<1, 2, 3, 4>(FVector v1, FVector v2) {
  return vextq_f32(v1, v2, 1);
}

template<>
inline Vector VectorPermute<2, 3, 4, 5>(FVector v1, FVector v2) {
  return vextq_f32(v1, v2, 2);
}

template<>
inline Vector VectorPermute<3, 4, 5, 6>(FVector v1, FVector v2) {
  return vextq_f32(v1, v2, 3);
}
#endif

Vector VectorSelectControl(uint32_t index0, uint32_t index1, uint32_t index2,
                           uint32_t index3);
Vector VectorSelect(FVector v1, FVector v2, FVector control);
Vector VectorMergeXY(FVector v1, FVector v2);
Vector VectorMergeZW(FVector v1, FVector v2);

Vector VectorEqual(FVector v1, FVector v2);
Vector VectorEqualInt(FVector v1, FVector v2);
Vector VectorNearEqual(FVector v1, FVector v2, FVector epsilon);
Vector VectorNotEqual(FVector v1, FVector v2);
Vector VectorIsNaN(FVector v);
Vector VectorIsInfinite(FVector v);
Vector VectorNegate(FVector v);
Vector VectorAdd(FVector v1, FVector v2);
Vector VectorSubtract(FVector v1, FVector v2);
Vector VectorMultiply(FVector v1, FVector v2);
Vector VectorMultiplyAdd(FVector v1, FVector v2, FVector v3);
Vector VectorDivide(FVector v1, FVector v2);
Vector VectorScale(FVector v, float scale);
Vector VectorNegativeMultiplySubtract(FVector v1, FVector v2, FVector v3);
Vector VectorReciprocalEst(FVector v);
Vector VectorReciprocal(FVector v);
Vector VectorSqrtEst(FVector v);
Vector VectorSqrt(FVector v);
Vector VectorReciprocalSqrtEst(FVector v);
Vector VectorReciprocalSqrt(FVector v);

bool Vector3Equal(FVector v1, FVector v2);
bool Vector3EqualInt(FVector v1, FVector v2);
bool Vector3NearEqual(FVector v1, FVector v2, FVector epsilon);
bool Vector3NotEqual(FVector v1, FVector v2);
bool Vector3IsNaN(FVector v);
bool Vector3IsInfinite(FVector v);
Vector Vector3Dot(FVector v1, FVector v2);
Vector Vector3Cross(FVector v1, FVector v2);
Vector Vector3LengthSq(FVector v);
Vector Vector3ReciprocalLengthEst(FVector v);
Vector Vector3ReciprocalLength(FVector v);
Vector Vector3LengthEst(FVector v);
Vector Vector3Length(FVector v);
Vector Vector3NormalizeEst(FVector v);
Vector Vector3Normalize(FVector v);
Vector Vector3Reflect(FVector incident, FVector normal);

bool Vector4Equal(FVector v1, FVector v2);
bool Vector4EqualInt(FVector v1, FVector v2);
bool Vector4NearEqual(FVector v1, FVector v2, FVector epsilon);
bool Vector4NotEqual(FVector v1, FVector v2);
bool Vector4IsNaN(FVector v);
bool Vector4IsInfinite(FVector v);
Vector Vector4Dot(FVector v1, FVector v2);
Vector Vector4Cross(FVector v1, FVector v2, FVector v3);
Vector Vector4LengthSq(FVector v);
Vector Vector4ReciprocalLengthEst(FVector v);
Vector Vector4ReciprocalLength(FVector v);
Vector Vector4LengthEst(FVector v);
Vector Vector4Length(FVector v);
Vector Vector4NormalizeEst(FVector v);
Vector Vector4Normalize(FVector v);
Vector Vector4Reflect(FVector incident, FVector normal);
}
}

#include "global_consts.h"

#if !NO_INLINE
#include "vectors_inline.h"
#endif
#endif