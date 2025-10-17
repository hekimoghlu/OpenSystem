/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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
#ifndef AOM_AV1_ENCODER_RANDOM_H_
#define AOM_AV1_ENCODER_RANDOM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Advance the generator to its next state, and generate the next 32-bit output.
// Note that the low bits of this output are comparatively low-quality, so users
// of this function should ensure that the high bits factor through to their
// outputs.
static inline uint32_t lcg_next(uint32_t *state) {
  *state = (uint32_t)(*state * 1103515245ULL + 12345);
  return *state;
}

// Generate a random number in the range [0, 32768).
static inline uint32_t lcg_rand16(uint32_t *state) {
  return (lcg_next(state) / 65536) % 32768;
}

// Generate a random number in the range [0, n)
// This is implemented as (rand() * n) / <range of RNG> rather than
// rand() % n, for a few reasons: This implementation is faster and less biased,
// and if is a power of 2, this uses the higher-quality top bits from the RNG
// output rather than the lower-quality bottom bits.
static inline uint32_t lcg_randint(uint32_t *state, uint32_t n) {
  uint64_t v = ((uint64_t)lcg_next(state) * n) >> 32;
  return (uint32_t)v;
}

// Generate a random number in the range [lo, hi)
static inline uint32_t lcg_randrange(uint32_t *state, uint32_t lo,
                                     uint32_t hi) {
  assert(lo < hi);
  return lo + lcg_randint(state, hi - lo);
}

// Pick k distinct numbers from the set {0, ..., n-1}
// All possible sets of k numbers, and all possible orderings of those numbers,
// are equally likely.
//
// Note: The algorithm used here uses resampling to avoid choosing repeated
// values. This works well as long as n >> k, but can potentially lead to many
// resampling attempts if n is equal to or only slightly larger than k.
static inline void lcg_pick(int n, int k, int *out, unsigned int *seed) {
  assert(0 <= k && k <= n);
  for (int i = 0; i < k; i++) {
    int v;

  // Inner resampling loop
  // We have to use a goto here because C does not have a multi-level continue
  // statement
  resample:
    v = (int)lcg_randint(seed, n);
    for (int j = 0; j < i; j++) {
      if (v == out[j]) {
        // Repeated v, resample
        goto resample;
      }
    }

    // New v, accept
    out[i] = v;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_RANDOM_H_
