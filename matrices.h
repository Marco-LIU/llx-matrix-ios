//
// Created by Marco.LIU on 16/8/16.
// Copyright (c) 2016 LLX. All rights reserved.
//

#ifndef _LLX_UTIL_MATRIX_MATRICES_H_
#define _LLX_UTIL_MATRIX_MATRICES_H_

#include "vectors.h"

// column-major matrices
namespace LLX {
namespace util {
struct Matrix;

#if USE_ARM_NEON && defined(__aarch64__)
typedef const Matrix FMatrix;
#else
typedef const Matrix& FMatrix;
#endif
typedef const Matrix& CMatrix;

struct __attribute__((aligned(16))) Matrix {
#if USE_ARM_NEON
  std::array<Vector, 4> c;
#else
  union {
    std::array<Vector, 4> c;
    float m[4][4];
    struct {
      float _00, _10, _20, _30;      // column 0
      float _01, _11, _21, _31;      // column 1
      float _02, _12, _22, _32;      // column 2
      float _03, _13, _23, _33;      // column 3
    };
  };
#endif
  Matrix() = default;
  Matrix(FVector c0, FVector c1, FVector c2, FVector c3) {
    c[0] = c0;
    c[1] = c1;
    c[2] = c2;
    c[3] = c3;
  }
  Matrix(float m00, float m10, float m20, float m30,
         float m01, float m11, float m21, float m31,
         float m02, float m12, float m22, float m32,
         float m03, float m13, float m23, float m33);

  Matrix& operator= (const Matrix& o) {
    if (this != &o) {
      c[0] = o.c[0];
      c[1] = o.c[1];
      c[2] = o.c[2];
      c[3] = o.c[3];
    }
    return *this;
  }

#if !USE_ARM_NEON
  float operator() (size_t column, size_t row) const { return m[column][row]; }
  float& operator() (size_t column, size_t row) { return m[column][row]; }
#endif

  Matrix operator+ () const { return *this; }
  Matrix operator- () const;

  Matrix& operator+= (FMatrix m);
  Matrix& operator-= (FMatrix m);
  Matrix& operator*= (FMatrix m);
  Matrix& operator*= (float s);
  Matrix& operator/= (float s);

  Matrix operator+ (FMatrix m) const;
  Matrix operator- (FMatrix m) const;
  Matrix operator* (FMatrix m) const;
  Matrix operator* (float s) const;
  Matrix operator/ (float s) const;

  //operator const float*() const { return reinterpret_cast<const float*>(this); }

  friend Matrix operator* (float s, FMatrix m);
};

bool MatrixIsNaN(FMatrix m);
bool MatrixIsInfinite(FMatrix m);
bool MatrixIsIdentity(FMatrix m);
Matrix MatrixMultiply(FMatrix m1, CMatrix m2);
Matrix MatrixMultiplyTranspose(FMatrix m1, FMatrix m2);
Matrix MatrixTranspose(FMatrix m);
Matrix MatrixInverse(Vector* determinant, FMatrix m);
Vector MatrixDeterminant(FMatrix m);
Matrix MatrixIdentity();
Matrix MatrixTranslate(float offset_x, float offset_y, float offset_z);
Matrix MatrixTranslate(FVector offset);
Matrix MatrixScale(float scale_x, float scale_y, float scale_z);
Matrix MatrixScale(FVector scale);
Matrix MatrixRotateQuaternion(FVector quaternion);

// 坐标系变换矩阵
// base_x, base_y, base_z必须为新坐标系的三个轴(标准正交基)
// translate为新坐标系原点在原坐标系的位置
Matrix MatrixSpaceTransform(FVector base_x, FVector base_y, FVector base_z,
                            GVector translate = Vector());

Matrix MatrixLookAt(FVector look_pos, FVector eye_pos, FVector up_dir);
Matrix MatrixLookAt(FVector look_pos);
Matrix MatrixLookTo(FVector look_dir, FVector eye_pos, FVector up_dir);
Matrix MatrixLookTo(FVector look_dir);

Matrix MatrixPerspective(float fov_y, float aspect_ratio, float near_z,
                         float far_z);
// return m * v
Vector VectorTransform(FMatrix m, FVector v);
}
}

#if !NO_INLINE
#include "matrices_inline.h"
#endif
#endif