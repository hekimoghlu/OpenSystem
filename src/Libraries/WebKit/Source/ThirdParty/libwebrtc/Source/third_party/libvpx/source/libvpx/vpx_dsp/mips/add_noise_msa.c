/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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
#include <stdlib.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/macros_msa.h"

void vpx_plane_add_noise_msa(uint8_t *start_ptr, const int8_t *noise,
                             int blackclamp, int whiteclamp, int width,
                             int height, int32_t pitch) {
  int i, j;
  v16u8 pos0, pos1, ref0, ref1;
  v16i8 black_clamp, white_clamp, both_clamp;

  black_clamp = __msa_fill_b(blackclamp);
  white_clamp = __msa_fill_b(whiteclamp);
  both_clamp = black_clamp + white_clamp;
  both_clamp = -both_clamp;

  for (i = 0; i < height / 2; ++i) {
    uint8_t *pos0_ptr = start_ptr + (2 * i) * pitch;
    const int8_t *ref0_ptr = noise + (rand() & 0xff);
    uint8_t *pos1_ptr = start_ptr + (2 * i + 1) * pitch;
    const int8_t *ref1_ptr = noise + (rand() & 0xff);
    for (j = width / 16; j--;) {
      pos0 = LD_UB(pos0_ptr);
      ref0 = LD_UB(ref0_ptr);
      pos1 = LD_UB(pos1_ptr);
      ref1 = LD_UB(ref1_ptr);
      pos0 = __msa_subsus_u_b(pos0, black_clamp);
      pos1 = __msa_subsus_u_b(pos1, black_clamp);
      pos0 = __msa_subsus_u_b(pos0, both_clamp);
      pos1 = __msa_subsus_u_b(pos1, both_clamp);
      pos0 = __msa_subsus_u_b(pos0, white_clamp);
      pos1 = __msa_subsus_u_b(pos1, white_clamp);
      pos0 += ref0;
      ST_UB(pos0, pos0_ptr);
      pos1 += ref1;
      ST_UB(pos1, pos1_ptr);
      pos0_ptr += 16;
      pos1_ptr += 16;
      ref0_ptr += 16;
      ref1_ptr += 16;
    }
  }
}
