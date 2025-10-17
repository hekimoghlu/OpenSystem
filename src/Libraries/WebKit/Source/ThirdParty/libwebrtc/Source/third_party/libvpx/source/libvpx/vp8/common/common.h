/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#ifndef VPX_VP8_COMMON_COMMON_H_
#define VPX_VP8_COMMON_COMMON_H_

#include <assert.h>

/* Interface header for common constant data structures and lookup tables */

#include "vpx_mem/vpx_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Only need this for fixed-size arrays, for structs just assign. */

#define vp8_copy(Dest, Src)              \
  do {                                   \
    assert(sizeof(Dest) == sizeof(Src)); \
    memcpy(Dest, Src, sizeof(Src));      \
  } while (0)

/* Use this for variably-sized arrays. */

#define vp8_copy_array(Dest, Src, N)           \
  do {                                         \
    assert(sizeof(*(Dest)) == sizeof(*(Src))); \
    memcpy(Dest, Src, (N) * sizeof(*(Src)));   \
  } while (0)

#define vp8_zero(Dest) memset(&(Dest), 0, sizeof(Dest))

#define vp8_zero_array(Dest, N) memset(Dest, 0, (N) * sizeof(*(Dest)))

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_COMMON_H_
