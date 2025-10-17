/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#include <stdio.h>

#include "./vpx_config.h"
#include "./vp9_rtcd.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_idct.h"
#include "vpx_dsp/mips/inv_txfm_dspr2.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_ports/mem.h"

#if HAVE_DSPR2
void vp9_iht16x16_256_add_dspr2(const int16_t *input, uint8_t *dest, int pitch,
                                int tx_type) {
  int i, j;
  DECLARE_ALIGNED(32, int16_t, out[16 * 16]);
  int16_t *outptr = out;
  int16_t temp_out[16];
  uint32_t pos = 45;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp    %[pos],    1    \n\t" : : [pos] "r"(pos));

  switch (tx_type) {
    case DCT_DCT:  // DCT in both horizontal and vertical
      idct16_rows_dspr2(input, outptr, 16);
      idct16_cols_add_blk_dspr2(out, dest, pitch);
      break;
    case ADST_DCT:  // ADST in vertical, DCT in horizontal
      idct16_rows_dspr2(input, outptr, 16);

      outptr = out;

      for (i = 0; i < 16; ++i) {
        iadst16_dspr2(outptr, temp_out);

        for (j = 0; j < 16; ++j)
          dest[j * pitch + i] = clip_pixel(ROUND_POWER_OF_TWO(temp_out[j], 6) +
                                           dest[j * pitch + i]);
        outptr += 16;
      }
      break;
    case DCT_ADST:  // DCT in vertical, ADST in horizontal
    {
      int16_t temp_in[16 * 16];

      for (i = 0; i < 16; ++i) {
        /* prefetch row */
        prefetch_load((const uint8_t *)(input + 16));

        iadst16_dspr2(input, outptr);
        input += 16;
        outptr += 16;
      }

      for (i = 0; i < 16; ++i)
        for (j = 0; j < 16; ++j) temp_in[j * 16 + i] = out[i * 16 + j];

      idct16_cols_add_blk_dspr2(temp_in, dest, pitch);
      break;
    }
    case ADST_ADST:  // ADST in both directions
    {
      int16_t temp_in[16];

      for (i = 0; i < 16; ++i) {
        /* prefetch row */
        prefetch_load((const uint8_t *)(input + 16));

        iadst16_dspr2(input, outptr);
        input += 16;
        outptr += 16;
      }

      for (i = 0; i < 16; ++i) {
        for (j = 0; j < 16; ++j) temp_in[j] = out[j * 16 + i];
        iadst16_dspr2(temp_in, temp_out);
        for (j = 0; j < 16; ++j)
          dest[j * pitch + i] = clip_pixel(ROUND_POWER_OF_TWO(temp_out[j], 6) +
                                           dest[j * pitch + i]);
      }
      break;
    }
    default: printf("vp9_short_iht16x16_add_dspr2 : Invalid tx_type\n"); break;
  }
}
#endif  // #if HAVE_DSPR2
