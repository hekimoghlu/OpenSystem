/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
#ifndef VPX_VP8_COMMON_INVTRANS_H_
#define VPX_VP8_COMMON_INVTRANS_H_

#include "./vpx_config.h"
#include "vp8_rtcd.h"
#include "blockd.h"
#include "onyxc_int.h"

#if CONFIG_MULTITHREAD
#include "vpx_mem/vpx_mem.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

static void eob_adjust(char *eobs, short *diff) {
  /* eob adjust.... the idct can only skip if both the dc and eob are zero */
  int js;
  for (js = 0; js < 16; ++js) {
    if ((eobs[js] == 0) && (diff[0] != 0)) eobs[js]++;
    diff += 16;
  }
}

static INLINE void vp8_inverse_transform_mby(MACROBLOCKD *xd) {
  short *DQC = xd->dequant_y1;

  if (xd->mode_info_context->mbmi.mode != SPLITMV) {
    /* do 2nd order transform on the dc block */
    if (xd->eobs[24] > 1) {
      vp8_short_inv_walsh4x4(&xd->block[24].dqcoeff[0], xd->qcoeff);
    } else {
      vp8_short_inv_walsh4x4_1(&xd->block[24].dqcoeff[0], xd->qcoeff);
    }
    eob_adjust(xd->eobs, xd->qcoeff);

    DQC = xd->dequant_y1_dc;
  }
  vp8_dequant_idct_add_y_block(xd->qcoeff, DQC, xd->dst.y_buffer,
                               xd->dst.y_stride, xd->eobs);
}
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_INVTRANS_H_
