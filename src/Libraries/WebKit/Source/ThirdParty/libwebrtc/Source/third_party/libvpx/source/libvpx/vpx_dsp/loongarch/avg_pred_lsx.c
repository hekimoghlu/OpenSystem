/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#include <assert.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_util/loongson_intrinsics.h"

void vpx_comp_avg_pred_lsx(uint8_t *comp_pred, const uint8_t *pred, int width,
                           int height, const uint8_t *ref, int ref_stride) {
  // width > 8 || width == 8 || width == 4
  if (width > 8) {
    int i, j;
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; j += 16) {
        __m128i p, r, avg;

        p = __lsx_vld(pred + j, 0);
        r = __lsx_vld(ref + j, 0);
        avg = __lsx_vavgr_bu(p, r);
        __lsx_vst(avg, comp_pred + j, 0);
      }
      comp_pred += width;
      pred += width;
      ref += ref_stride;
    }
  } else if (width == 8) {
    int i = height * width;
    do {
      __m128i p, r, r_0, r_1;

      p = __lsx_vld(pred, 0);
      r_0 = __lsx_vld(ref, 0);
      ref += ref_stride;
      r_1 = __lsx_vld(ref, 0);
      ref += ref_stride;
      r = __lsx_vilvl_d(r_1, r_0);
      r = __lsx_vavgr_bu(p, r);

      __lsx_vst(r, comp_pred, 0);

      pred += 16;
      comp_pred += 16;
      i -= 16;
    } while (i);
  } else {  // width = 4
    int i = height * width;
    assert(width == 4);
    do {
      __m128i p, r, r_0, r_1, r_2, r_3;
      p = __lsx_vld(pred, 0);

      if (width == ref_stride) {
        r = __lsx_vld(ref, 0);
        ref += 16;
      } else {
        r_0 = __lsx_vld(ref, 0);
        ref += ref_stride;
        r_1 = __lsx_vld(ref, 0);
        ref += ref_stride;
        r_2 = __lsx_vld(ref, 0);
        ref += ref_stride;
        r_3 = __lsx_vld(ref, 0);
        ref += ref_stride;
        DUP2_ARG2(__lsx_vilvl_w, r_1, r_0, r_3, r_2, r_0, r_2);
        r = __lsx_vilvl_d(r_2, r_0);
      }
      r = __lsx_vavgr_bu(p, r);

      __lsx_vst(r, comp_pred, 0);
      comp_pred += 16;
      pred += 16;
      i -= 16;
    } while (i);
  }
}
