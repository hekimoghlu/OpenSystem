/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
#ifndef VPX_VPX_DSP_FWD_TXFM_H_
#define VPX_VPX_DSP_FWD_TXFM_H_

#include "vpx_dsp/txfm_common.h"

static INLINE tran_high_t fdct_round_shift(tran_high_t input) {
  tran_high_t rv = ROUND_POWER_OF_TWO(input, DCT_CONST_BITS);
  // TODO(debargha, peter.derivaz): Find new bounds for this assert
  // and make the bounds consts.
  // assert(INT16_MIN <= rv && rv <= INT16_MAX);
  return rv;
}

void vpx_fdct32(const tran_high_t *input, tran_high_t *output, int round);
#endif  // VPX_VPX_DSP_FWD_TXFM_H_
