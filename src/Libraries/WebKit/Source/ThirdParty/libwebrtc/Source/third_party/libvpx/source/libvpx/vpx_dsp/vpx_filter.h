/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#ifndef VPX_VPX_DSP_VPX_FILTER_H_
#define VPX_VPX_DSP_VPX_FILTER_H_

#include <assert.h>
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FILTER_BITS 7

#define SUBPEL_BITS 4
#define SUBPEL_MASK ((1 << SUBPEL_BITS) - 1)
#define SUBPEL_SHIFTS (1 << SUBPEL_BITS)
#define SUBPEL_TAPS 8

typedef int16_t InterpKernel[SUBPEL_TAPS];

static INLINE int vpx_get_filter_taps(const int16_t *const filter) {
  if (filter[0] | filter[7]) {
    return 8;
  }
  if (filter[1] | filter[6]) {
    return 6;
  }
  if (filter[2] | filter[5]) {
    return 4;
  }
  return 2;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_VPX_FILTER_H_
