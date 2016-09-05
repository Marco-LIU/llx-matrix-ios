//
// Created by Marco.LIU on 16/8/19.
// Copyright (c) 2016 LLX. All rights reserved.
//

#if !NO_INLINE
#include "vectors.h"
#endif

#include "global_consts.h"

namespace LLX {
namespace util {
const VectorF32 kZeroVector = {0.0f, 0.0f, 0.0f, 0.0f};
const VectorF32 kEpsilon =
    {1.192092896e-7f, 1.192092896e-7f, 1.192092896e-7f, 1.192092896e-7f};
const VectorI32 kInfinity =
    {0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000};
const VectorI32 kQNaN =
    {0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000};
const VectorI32 kQNaNTest =
    {0x007FFFFF, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF};
const VectorI32 kAbsMask =
    {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
const VectorU32 kFlipY = {0, 0x80000000, 0, 0};
const VectorF32 kIdentityX = {1.0f, 0.0f, 0.0f, 0.0f};
const VectorF32 kIdentityY = {0.0f, 1.0f, 0.0f, 0.0f};
const VectorF32 kIdentityZ = {0.0f, 0.0f, 1.0f, 0.0f};
const VectorF32 kIdentityW = {0.0f, 0.0f, 0.0f, 1.0f};
const VectorF32 kNegIdentityX = {-1.0f,0.0f, 0.0f, 0.0f};
const VectorF32 kNegIdentityY = {0.0f,-1.0f, 0.0f, 0.0f};
const VectorF32 kNegIdentityZ = {0.0f, 0.0f,-1.0f, 0.0f};
const VectorF32 kNegIdentityW = {0.0f, 0.0f, 0.0f,-1.0f};
const VectorU32 kMask3 =
    {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000};
const VectorU32 kMaskX =
    {0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000};
const VectorU32 kMaskY =
    {0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000};
const VectorU32 kMaskZ =
    {0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000};
const VectorU32 kMaskW =
    {0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF};

const VectorU32 kSelect0101 =
    {kSelect0, kSelect1, kSelect0, kSelect1};
const VectorU32 kSelect1010 =
    {kSelect1, kSelect0, kSelect1, kSelect0};
const VectorU32 kSelect1000 =
    {kSelect1, kSelect0, kSelect0, kSelect0};
const VectorU32 kSelect1100 =
    {kSelect1, kSelect1, kSelect0, kSelect0};
const VectorU32 kSelect1110 =
    {kSelect1, kSelect1, kSelect1, kSelect0};
const VectorU32 kSelect1011 =
    {kSelect1, kSelect0, kSelect1, kSelect1};
const VectorU32 kOneHalfMinusEpsilon =
    {0x3EFFFFFD, 0x3EFFFFFD, 0x3EFFFFFD, 0x3EFFFFFD};

}
}