//
// Created by Marco.LIU on 16/8/18.
// Copyright (c) 2016 LLX. All rights reserved.
//

#ifndef _LLX_UTIL_MATRIX_MATRIX_CONFIG_H_
#define _LLX_UTIL_MATRIX_MATRIX_CONFIG_H_

#include <TargetConditionals.h>

#define NO_INLINE 0

#if defined(TARGET_OS_SIMULATOR) && TARGET_OS_SIMULATOR == 0
#define USE_ARM_NEON 1
#include <arm_neon.h>
#else
#define USE_ARM_NEON 0
#endif

#if NO_INLINE
#define LLX_INLINE
#else
#define LLX_INLINE inline
#endif

#endif