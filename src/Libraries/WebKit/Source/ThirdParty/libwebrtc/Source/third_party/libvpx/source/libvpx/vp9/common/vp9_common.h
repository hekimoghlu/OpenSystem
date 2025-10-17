/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
#ifndef VPX_VP9_COMMON_VP9_COMMON_H_
#define VPX_VP9_COMMON_VP9_COMMON_H_

/* Interface header for common constant data structures and lookup tables */

#include <assert.h>

#include "./vpx_config.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/bitops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Only need this for fixed-size arrays, for structs just assign.
#define vp9_copy(dest, src)              \
  do {                                   \
    assert(sizeof(dest) == sizeof(src)); \
    memcpy(dest, src, sizeof(src));      \
  } while (0)

// Use this for variably-sized arrays.
#define vp9_copy_array(dest, src, n)           \
  {                                            \
    assert(sizeof(*(dest)) == sizeof(*(src))); \
    memcpy(dest, src, (n) * sizeof(*(src)));   \
  }

#define vp9_zero(dest) memset(&(dest), 0, sizeof(dest))
#define vp9_zero_array(dest, n) memset(dest, 0, (n) * sizeof(*(dest)))

static INLINE int get_unsigned_bits(unsigned int num_values) {
  return num_values > 0 ? get_msb(num_values) + 1 : 0;
}

#define VP9_SYNC_CODE_0 0x49
#define VP9_SYNC_CODE_1 0x83
#define VP9_SYNC_CODE_2 0x42

#define VP9_FRAME_MARKER 0x2

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_COMMON_H_
