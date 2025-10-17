/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#include "segmentation.h"
#include "vpx_mem/vpx_mem.h"

void vp8_update_gf_usage_maps(VP8_COMP *cpi, VP8_COMMON *cm, MACROBLOCK *x) {
  int mb_row, mb_col;

  MODE_INFO *this_mb_mode_info = cm->mi;

  x->gf_active_ptr = (signed char *)cpi->gf_active_flags;

  if ((cm->frame_type == KEY_FRAME) || (cm->refresh_golden_frame)) {
    /* Reset Gf usage monitors */
    memset(cpi->gf_active_flags, 1, (cm->mb_rows * cm->mb_cols));
    cpi->gf_active_count = cm->mb_rows * cm->mb_cols;
  } else {
    /* for each macroblock row in image */
    for (mb_row = 0; mb_row < cm->mb_rows; ++mb_row) {
      /* for each macroblock col in image */
      for (mb_col = 0; mb_col < cm->mb_cols; ++mb_col) {
        /* If using golden then set GF active flag if not already set.
         * If using last frame 0,0 mode then leave flag as it is
         * else if using non 0,0 motion or intra modes then clear
         * flag if it is currently set
         */
        if ((this_mb_mode_info->mbmi.ref_frame == GOLDEN_FRAME) ||
            (this_mb_mode_info->mbmi.ref_frame == ALTREF_FRAME)) {
          if (*(x->gf_active_ptr) == 0) {
            *(x->gf_active_ptr) = 1;
            cpi->gf_active_count++;
          }
        } else if ((this_mb_mode_info->mbmi.mode != ZEROMV) &&
                   *(x->gf_active_ptr)) {
          *(x->gf_active_ptr) = 0;
          cpi->gf_active_count--;
        }

        x->gf_active_ptr++;  /* Step onto next entry */
        this_mb_mode_info++; /* skip to next mb */
      }

      /* this is to account for the border */
      this_mb_mode_info++;
    }
  }
}
