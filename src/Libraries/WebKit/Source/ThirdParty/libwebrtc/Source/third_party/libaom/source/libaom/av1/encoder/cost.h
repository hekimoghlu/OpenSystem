/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#ifndef AOM_AV1_ENCODER_COST_H_
#define AOM_AV1_ENCODER_COST_H_

#include "aom_dsp/prob.h"
#include "aom/aom_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const uint16_t av1_prob_cost[128];

// The factor to scale from cost in bits to cost in av1_prob_cost units.
#define AV1_PROB_COST_SHIFT 9

// Cost of coding an n bit literal, using 128 (i.e. 50%) probability
// for each bit.
#define av1_cost_literal(n) ((n) * (1 << AV1_PROB_COST_SHIFT))

// Calculate the cost of a symbol with probability p15 / 2^15
static inline int av1_cost_symbol(aom_cdf_prob p15) {
  // p15 can be out of range [1, CDF_PROB_TOP - 1]. Clamping it, so that the
  // following cost calculation works correctly. Otherwise, if p15 =
  // CDF_PROB_TOP, shift would be -1, and "p15 << shift" would be wrong.
  p15 = (aom_cdf_prob)clamp(p15, 1, CDF_PROB_TOP - 1);
  assert(0 < p15 && p15 < CDF_PROB_TOP);
  const int shift = CDF_PROB_BITS - 1 - get_msb(p15);
  const int prob = get_prob(p15 << shift, CDF_PROB_TOP);
  assert(prob >= 128);
  return av1_prob_cost[prob - 128] + av1_cost_literal(shift);
}

void av1_cost_tokens_from_cdf(int *costs, const aom_cdf_prob *cdf,
                              const int *inv_map);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_COST_H_
