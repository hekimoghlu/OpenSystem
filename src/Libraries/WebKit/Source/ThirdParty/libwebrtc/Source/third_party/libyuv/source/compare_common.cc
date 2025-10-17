/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#include "libyuv/basic_types.h"

#include "libyuv/compare_row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Hakmem method for hamming distance.
uint32_t HammingDistance_C(const uint8_t* src_a,
                           const uint8_t* src_b,
                           int count) {
  uint32_t diff = 0u;

  int i;
  for (i = 0; i < count - 3; i += 4) {
    uint32_t x = *((const uint32_t*)src_a) ^ *((const uint32_t*)src_b);
    uint32_t u = x - ((x >> 1) & 0x55555555);
    u = ((u >> 2) & 0x33333333) + (u & 0x33333333);
    diff += ((((u + (u >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24);
    src_a += 4;
    src_b += 4;
  }

  for (; i < count; ++i) {
    uint32_t x = *src_a ^ *src_b;
    uint32_t u = x - ((x >> 1) & 0x55);
    u = ((u >> 2) & 0x33) + (u & 0x33);
    diff += (u + (u >> 4)) & 0x0f;
    src_a += 1;
    src_b += 1;
  }

  return diff;
}

uint32_t SumSquareError_C(const uint8_t* src_a,
                          const uint8_t* src_b,
                          int count) {
  uint32_t sse = 0u;
  int i;
  for (i = 0; i < count; ++i) {
    int diff = src_a[i] - src_b[i];
    sse += (uint32_t)(diff * diff);
  }
  return sse;
}

// hash seed of 5381 recommended.
// Internal C version of HashDjb2 with int sized count for efficiency.
uint32_t HashDjb2_C(const uint8_t* src, int count, uint32_t seed) {
  uint32_t hash = seed;
  int i;
  for (i = 0; i < count; ++i) {
    hash += (hash << 5) + src[i];
  }
  return hash;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
