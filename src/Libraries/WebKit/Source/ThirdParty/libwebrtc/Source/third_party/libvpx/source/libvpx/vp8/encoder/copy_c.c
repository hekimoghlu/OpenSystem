/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#include <string.h>

#include "./vp8_rtcd.h"
#include "vpx/vpx_integer.h"

/* Copy 2 macroblocks to a buffer */
void vp8_copy32xn_c(const unsigned char *src_ptr, int src_stride,
                    unsigned char *dst_ptr, int dst_stride, int height) {
  int r;

  for (r = 0; r < height; ++r) {
    memcpy(dst_ptr, src_ptr, 32);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}
