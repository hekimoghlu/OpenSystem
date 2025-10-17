/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#ifndef AOM_AOM_DSP_AOM_SIMD_H_
#define AOM_AOM_DSP_AOM_SIMD_H_

#include <stdint.h>

#if defined(_WIN32)
#include <intrin.h>
#endif

#include "config/aom_config.h"

#include "aom_dsp/aom_simd_inline.h"

#define SIMD_CHECK 1  // Sanity checks in C equivalents

// VS compiling for 32 bit targets does not support vector types in
// structs as arguments, which makes the v256 type of the intrinsics
// hard to support, so optimizations for this target are disabled.
#if HAVE_SSE2 && (defined(_WIN64) || !defined(_MSC_VER) || defined(__clang__))
#include "simd/v256_intrinsics_x86.h"
#else
#include "simd/v256_intrinsics.h"
#endif

#endif  // AOM_AOM_DSP_AOM_SIMD_H_
