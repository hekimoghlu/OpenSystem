/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#ifndef VPX_VP9_ENCODER_VP9_COST_H_
#define VPX_VP9_ENCODER_VP9_COST_H_

#include "vpx_dsp/prob.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const uint16_t vp9_prob_cost[256];

// The factor to scale from cost in bits to cost in vp9_prob_cost units.
#define VP9_PROB_COST_SHIFT 9

#define vp9_cost_zero(prob) (vp9_prob_cost[prob])

#define vp9_cost_one(prob) vp9_cost_zero(256 - (prob))

#define vp9_cost_bit(prob, bit) vp9_cost_zero((bit) ? 256 - (prob) : (prob))

static INLINE uint64_t cost_branch256(const unsigned int ct[2], vpx_prob p) {
  return (uint64_t)ct[0] * vp9_cost_zero(p) + (uint64_t)ct[1] * vp9_cost_one(p);
}

static INLINE int treed_cost(vpx_tree tree, const vpx_prob *probs, int bits,
                             int len) {
  int cost = 0;
  vpx_tree_index i = 0;

  do {
    const int bit = (bits >> --len) & 1;
    cost += vp9_cost_bit(probs[i >> 1], bit);
    i = tree[i + bit];
  } while (len);

  return cost;
}

void vp9_cost_tokens(int *costs, const vpx_prob *probs, vpx_tree tree);
void vp9_cost_tokens_skip(int *costs, const vpx_prob *probs, vpx_tree tree);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_COST_H_
