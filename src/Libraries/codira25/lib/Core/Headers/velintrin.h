/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef __VEL_INTRIN_H__
#define __VEL_INTRIN_H__

// Vector registers
typedef double __vr __attribute__((__vector_size__(2048)));

// Vector mask registers
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
// For C99
typedef _Bool __vm    __attribute__((ext_vector_type(256)));
typedef _Bool __vm256 __attribute__((ext_vector_type(256)));
typedef _Bool __vm512 __attribute__((ext_vector_type(512)));
#else
#ifdef __cplusplus
// For C++
typedef bool __vm    __attribute__((ext_vector_type(256)));
typedef bool __vm256 __attribute__((ext_vector_type(256)));
typedef bool __vm512 __attribute__((ext_vector_type(512)));
#else
#error need C++ or C99 to use vector intrinsics for VE
#endif
#endif

enum VShuffleCodes {
  VE_VSHUFFLE_YUYU = 0,
  VE_VSHUFFLE_YUYL = 1,
  VE_VSHUFFLE_YUZU = 2,
  VE_VSHUFFLE_YUZL = 3,
  VE_VSHUFFLE_YLYU = 4,
  VE_VSHUFFLE_YLYL = 5,
  VE_VSHUFFLE_YLZU = 6,
  VE_VSHUFFLE_YLZL = 7,
  VE_VSHUFFLE_ZUYU = 8,
  VE_VSHUFFLE_ZUYL = 9,
  VE_VSHUFFLE_ZUZU = 10,
  VE_VSHUFFLE_ZUZL = 11,
  VE_VSHUFFLE_ZLYU = 12,
  VE_VSHUFFLE_ZLYL = 13,
  VE_VSHUFFLE_ZLZU = 14,
  VE_VSHUFFLE_ZLZL = 15,
};

// Use generated intrinsic name definitions
#include <velintrin_gen.h>

// Use helper functions
#include <velintrin_approx.h>

// pack

#define _vel_pack_f32p __builtin_ve_vl_pack_f32p
#define _vel_pack_f32a __builtin_ve_vl_pack_f32a

static inline unsigned long int _vel_pack_i32(unsigned int a, unsigned int b) {
  return (((unsigned long int)a) << 32) | b;
}

#define _vel_extract_vm512u(vm) __builtin_ve_vl_extract_vm512u(vm)
#define _vel_extract_vm512l(vm) __builtin_ve_vl_extract_vm512l(vm)
#define _vel_insert_vm512u(vm512, vm) __builtin_ve_vl_insert_vm512u(vm512, vm)
#define _vel_insert_vm512l(vm512, vm) __builtin_ve_vl_insert_vm512l(vm512, vm)

#endif
