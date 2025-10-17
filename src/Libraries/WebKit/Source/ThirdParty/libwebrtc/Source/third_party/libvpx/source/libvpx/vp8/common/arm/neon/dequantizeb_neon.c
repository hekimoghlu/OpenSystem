/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#include <arm_neon.h>

#include "./vp8_rtcd.h"
#include "vp8/common/blockd.h"

void vp8_dequantize_b_neon(BLOCKD *d, short *DQC) {
  int16x8x2_t qQ, qDQC, qDQ;

  qQ = vld2q_s16(d->qcoeff);
  qDQC = vld2q_s16(DQC);

  qDQ.val[0] = vmulq_s16(qQ.val[0], qDQC.val[0]);
  qDQ.val[1] = vmulq_s16(qQ.val[1], qDQC.val[1]);

  vst2q_s16(d->dqcoeff, qDQ);
}
