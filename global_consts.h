//
// Created by Marco.LIU on 16/8/19.
// Copyright (c) 2016 LLX. All rights reserved.
//

#ifndef _LLX_UTIL_MATRIX_GLOBAL_CONSTS_H_
#define _LLX_UTIL_MATRIX_GLOBAL_CONSTS_H_

#include "matrix_config.h"

#include <math.h>

#if NO_INLINE
#include "vectors.h"
#endif

// PI
#define LLX_PI 3.14159265358979323846264338327950288
// 2PI
#define LLX_2PI (2 * LLX_PI)
// PI / 2
#define LLX_PI_2 1.57079632679489661923132169163975144
// PI / 4
#define LLX_PI_4 0.785398163397448309615660845819875721
// 1 / PI
#define LLX_1_PI 0.318309886183790671537767526745028724
// 1 / 2PI
#define LLX_1_2PI (0.5 * LLX_1_PI)
// 2 / PI
#define LLX_2_PI 0.636619772367581343075535053490057448
// 2 / sqrt(PI)
#define LLX_2_SQRTPI 1.12837916709551257389615890312154517

namespace LLX {
namespace util {
const uint32_t kSelect0          = 0x00000000;
const uint32_t kSelect1          = 0xFFFFFFFF;

const uint32_t kPermute0X        = 0;
const uint32_t kPermute0Y        = 1;
const uint32_t kPermute0Z        = 2;
const uint32_t kPermute0W        = 3;
const uint32_t kPermute1X        = 4;
const uint32_t kPermute1Y        = 5;
const uint32_t kPermute1Z        = 6;
const uint32_t kPermute1W        = 7;

const uint32_t kSwizzleX         = 0;
const uint32_t kSwizzleY         = 1;
const uint32_t kSwizzleZ         = 2;
const uint32_t kSwizzleW         = 3;

#ifndef LLX_GLOBAL_CONST
#define LLX_GLOBAL_CONST extern const __attribute__((weak))
#endif

// {0.0f, 0.0f, 0.0f, 0.0f}
LLX_GLOBAL_CONST VectorF32 kZeroVector;
// {1.192092896e-7f, 1.192092896e-7f, 1.192092896e-7f, 1.192092896e-7f};
LLX_GLOBAL_CONST VectorF32 kEpsilon;
// {0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000}
LLX_GLOBAL_CONST VectorI32 kInfinity;
// {0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000}
LLX_GLOBAL_CONST VectorI32 kQNaN;
// {0x007FFFFF, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF}
LLX_GLOBAL_CONST VectorI32 kQNaNTest;
// {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF}
LLX_GLOBAL_CONST VectorI32 kAbsMask;
// {0, 0x80000000, 0, 0}
LLX_GLOBAL_CONST VectorU32 kFlipY;
// {1.0f, 0.0f, 0.0f, 0.0f}
LLX_GLOBAL_CONST VectorF32 kIdentityX;
// {0.0f, 1.0f, 0.0f, 0.0f}
LLX_GLOBAL_CONST VectorF32 kIdentityY;
// {0.0f, 0.0f, 1.0f, 0.0f}
LLX_GLOBAL_CONST VectorF32 kIdentityZ;
// {0.0f, 0.0f, 0.0f, 1.0f}
LLX_GLOBAL_CONST VectorF32 kIdentityW;
// {-1.0f,0.0f, 0.0f, 0.0f}
LLX_GLOBAL_CONST VectorF32 kNegIdentityX;
// {0.0f,-1.0f, 0.0f, 0.0f}
LLX_GLOBAL_CONST VectorF32 kNegIdentityY;
// {0.0f, 0.0f,-1.0f, 0.0f}
LLX_GLOBAL_CONST VectorF32 kNegIdentityZ;
// {0.0f, 0.0f, 0.0f,-1.0f}
LLX_GLOBAL_CONST VectorF32 kNegIdentityW;
// {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000}
LLX_GLOBAL_CONST VectorU32 kMask3;
// {0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000}
LLX_GLOBAL_CONST VectorU32 kMaskX;
// {0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000}
LLX_GLOBAL_CONST VectorU32 kMaskY;
// {0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000}
LLX_GLOBAL_CONST VectorU32 kMaskZ;
// {0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF}
LLX_GLOBAL_CONST VectorU32 kMaskW;

// {SELECT_0, LLX_SELECT_1, SELECT_0, LLX_SELECT_1}
LLX_GLOBAL_CONST VectorU32 kSelect0101;
// {LLX_SELECT_1, SELECT_0, LLX_SELECT_1, SELECT_0}
LLX_GLOBAL_CONST VectorU32 kSelect1010;
// {LLX_SELECT_1, SELECT_0, SELECT_0, SELECT_0}
LLX_GLOBAL_CONST VectorU32 kSelect1000;
// {LLX_SELECT_1, LLX_SELECT_1, SELECT_0, SELECT_0}
LLX_GLOBAL_CONST VectorU32 kSelect1100;
// {LLX_SELECT_1, LLX_SELECT_1, LLX_SELECT_1, SELECT_0}
LLX_GLOBAL_CONST VectorU32 kSelect1110;
// {LLX_SELECT_1, SELECT_0, LLX_SELECT_1, LLX_SELECT_1}
LLX_GLOBAL_CONST VectorU32 kSelect1011;
// {0x3EFFFFFD, 0x3EFFFFFD, 0x3EFFFFFD, 0x3EFFFFFD}
LLX_GLOBAL_CONST VectorU32 kOneHalfMinusEpsilon;


}
}
#endif //_LLX_UTIL_MATRIX_GLOBAL_CONSTS_H_
