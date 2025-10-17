/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

#include "./vp9_rtcd.h"
#include "vp9/common/vp9_enums.h"
#include "vpx_dsp/mips/inv_txfm_msa.h"

void vp9_iht16x16_256_add_msa(const int16_t *input, uint8_t *dst,
                              int32_t dst_stride, int32_t tx_type) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, out[16 * 16]);
  int16_t *out_ptr = &out[0];

  switch (tx_type) {
    case DCT_DCT:
      /* transform rows */
      for (i = 0; i < 2; ++i) {
        /* process 16 * 8 block */
        vpx_idct16_1d_rows_msa((input + (i << 7)), (out_ptr + (i << 7)));
      }

      /* transform columns */
      for (i = 0; i < 2; ++i) {
        /* process 8 * 16 block */
        vpx_idct16_1d_columns_addblk_msa((out_ptr + (i << 3)), (dst + (i << 3)),
                                         dst_stride);
      }
      break;
    case ADST_DCT:
      /* transform rows */
      for (i = 0; i < 2; ++i) {
        /* process 16 * 8 block */
        vpx_idct16_1d_rows_msa((input + (i << 7)), (out_ptr + (i << 7)));
      }

      /* transform columns */
      for (i = 0; i < 2; ++i) {
        vpx_iadst16_1d_columns_addblk_msa((out_ptr + (i << 3)),
                                          (dst + (i << 3)), dst_stride);
      }
      break;
    case DCT_ADST:
      /* transform rows */
      for (i = 0; i < 2; ++i) {
        /* process 16 * 8 block */
        vpx_iadst16_1d_rows_msa((input + (i << 7)), (out_ptr + (i << 7)));
      }

      /* transform columns */
      for (i = 0; i < 2; ++i) {
        /* process 8 * 16 block */
        vpx_idct16_1d_columns_addblk_msa((out_ptr + (i << 3)), (dst + (i << 3)),
                                         dst_stride);
      }
      break;
    case ADST_ADST:
      /* transform rows */
      for (i = 0; i < 2; ++i) {
        /* process 16 * 8 block */
        vpx_iadst16_1d_rows_msa((input + (i << 7)), (out_ptr + (i << 7)));
      }

      /* transform columns */
      for (i = 0; i < 2; ++i) {
        vpx_iadst16_1d_columns_addblk_msa((out_ptr + (i << 3)),
                                          (dst + (i << 3)), dst_stride);
      }
      break;
    default: assert(0); break;
  }
}
