/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx_mem/vpx_mem.h"

#if HAVE_DSPR2
void vp8_dequant_idct_add_dspr2(short *input, short *dq, unsigned char *dest,
                                int stride) {
  int i;

  for (i = 0; i < 16; ++i) {
    input[i] = dq[i] * input[i];
  }

  vp8_short_idct4x4llm_dspr2(input, dest, stride, dest, stride);

  memset(input, 0, 32);
}

#endif
