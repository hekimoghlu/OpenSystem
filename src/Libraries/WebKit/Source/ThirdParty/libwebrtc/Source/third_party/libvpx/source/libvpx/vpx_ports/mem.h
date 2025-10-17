/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
#ifndef VPX_VPX_PORTS_MEM_H_
#define VPX_VPX_PORTS_MEM_H_

#include "vpx_config.h"
#include "vpx/vpx_integer.h"

#if defined(__GNUC__) || defined(__SUNPRO_C)
#define DECLARE_ALIGNED(n, typ, val) typ val __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define DECLARE_ALIGNED(n, typ, val) __declspec(align(n)) typ val
#else
#warning No alignment directives known for this compiler.
#define DECLARE_ALIGNED(n, typ, val) typ val
#endif

#if defined(__has_builtin)
#define VPX_HAS_BUILTIN(x) __has_builtin(x)
#else
#define VPX_HAS_BUILTIN(x) 0
#endif

#if !VPX_HAS_BUILTIN(__builtin_prefetch) && !defined(__GNUC__)
#define __builtin_prefetch(x)
#endif

/* Shift down with rounding */
#define ROUND_POWER_OF_TWO(value, n) (((value) + (1 << ((n)-1))) >> (n))
#define ROUND64_POWER_OF_TWO(value, n) (((value) + (1ULL << ((n)-1))) >> (n))

#define ALIGN_POWER_OF_TWO(value, n) \
  (((value) + ((1 << (n)) - 1)) & ~((1 << (n)) - 1))

#define CONVERT_TO_SHORTPTR(x) ((uint16_t *)(((uintptr_t)(x)) << 1))
#define CAST_TO_SHORTPTR(x) ((uint16_t *)((uintptr_t)(x)))
#if CONFIG_VP9_HIGHBITDEPTH
#define CONVERT_TO_BYTEPTR(x) ((uint8_t *)(((uintptr_t)(x)) >> 1))
#define CAST_TO_BYTEPTR(x) ((uint8_t *)((uintptr_t)(x)))
#endif  // CONFIG_VP9_HIGHBITDEPTH

#endif  // VPX_VPX_PORTS_MEM_H_
