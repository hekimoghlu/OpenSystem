/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
#include "vpx_dsp/mips/common_dspr2.h"

#if HAVE_DSPR2
uint8_t vpx_ff_cropTbl_a[256 + 2 * CROP_WIDTH];
uint8_t *vpx_ff_cropTbl;

void vpx_dsputil_static_init(void) {
  int i;

  for (i = 0; i < 256; i++) vpx_ff_cropTbl_a[i + CROP_WIDTH] = i;

  for (i = 0; i < CROP_WIDTH; i++) {
    vpx_ff_cropTbl_a[i] = 0;
    vpx_ff_cropTbl_a[i + CROP_WIDTH + 256] = 255;
  }

  vpx_ff_cropTbl = &vpx_ff_cropTbl_a[CROP_WIDTH];
}

#endif
