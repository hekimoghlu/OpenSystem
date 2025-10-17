/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#ifndef AOM_AOM_DSP_RECENTER_H_
#define AOM_AOM_DSP_RECENTER_H_

#include "config/aom_config.h"

#include "aom/aom_integer.h"

// Inverse recenters a non-negative literal v around a reference r
static inline uint16_t inv_recenter_nonneg(uint16_t r, uint16_t v) {
  if (v > (r << 1))
    return v;
  else if ((v & 1) == 0)
    return (v >> 1) + r;
  else
    return r - ((v + 1) >> 1);
}

// Inverse recenters a non-negative literal v in [0, n-1] around a
// reference r also in [0, n-1]
static inline uint16_t inv_recenter_finite_nonneg(uint16_t n, uint16_t r,
                                                  uint16_t v) {
  if ((r << 1) <= n) {
    return inv_recenter_nonneg(r, v);
  } else {
    return n - 1 - inv_recenter_nonneg(n - 1 - r, v);
  }
}

// Recenters a non-negative literal v around a reference r
static inline uint16_t recenter_nonneg(uint16_t r, uint16_t v) {
  if (v > (r << 1))
    return v;
  else if (v >= r)
    return ((v - r) << 1);
  else
    return ((r - v) << 1) - 1;
}

// Recenters a non-negative literal v in [0, n-1] around a
// reference r also in [0, n-1]
static inline uint16_t recenter_finite_nonneg(uint16_t n, uint16_t r,
                                              uint16_t v) {
  if ((r << 1) <= n) {
    return recenter_nonneg(r, v);
  } else {
    return recenter_nonneg(n - 1 - r, n - 1 - v);
  }
}

#endif  // AOM_AOM_DSP_RECENTER_H_
