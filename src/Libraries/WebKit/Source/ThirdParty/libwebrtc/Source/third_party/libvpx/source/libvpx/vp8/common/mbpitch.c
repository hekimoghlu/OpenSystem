/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#include "blockd.h"

void vp8_setup_block_dptrs(MACROBLOCKD *x) {
  int r, c;

  for (r = 0; r < 4; ++r) {
    for (c = 0; c < 4; ++c) {
      x->block[r * 4 + c].predictor = x->predictor + r * 4 * 16 + c * 4;
    }
  }

  for (r = 0; r < 2; ++r) {
    for (c = 0; c < 2; ++c) {
      x->block[16 + r * 2 + c].predictor =
          x->predictor + 256 + r * 4 * 8 + c * 4;
    }
  }

  for (r = 0; r < 2; ++r) {
    for (c = 0; c < 2; ++c) {
      x->block[20 + r * 2 + c].predictor =
          x->predictor + 320 + r * 4 * 8 + c * 4;
    }
  }

  for (r = 0; r < 25; ++r) {
    x->block[r].qcoeff = x->qcoeff + r * 16;
    x->block[r].dqcoeff = x->dqcoeff + r * 16;
    x->block[r].eob = x->eobs + r;
  }
}

void vp8_build_block_doffsets(MACROBLOCKD *x) {
  int block;

  for (block = 0; block < 16; ++block) /* y blocks */
  {
    x->block[block].offset =
        (block >> 2) * 4 * x->dst.y_stride + (block & 3) * 4;
  }

  for (block = 16; block < 20; ++block) /* U and V blocks */
  {
    x->block[block + 4].offset = x->block[block].offset =
        ((block - 16) >> 1) * 4 * x->dst.uv_stride + (block & 1) * 4;
  }
}
