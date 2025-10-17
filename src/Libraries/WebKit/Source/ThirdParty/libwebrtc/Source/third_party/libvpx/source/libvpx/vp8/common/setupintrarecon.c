/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "setupintrarecon.h"
#include "vpx_mem/vpx_mem.h"

void vp8_setup_intra_recon(YV12_BUFFER_CONFIG *ybf) {
  int i;

  /* set up frame new frame for intra coded blocks */
  memset(ybf->y_buffer - 1 - ybf->y_stride, 127, ybf->y_width + 5);
  for (i = 0; i < ybf->y_height; ++i) {
    ybf->y_buffer[ybf->y_stride * i - 1] = (unsigned char)129;
  }

  memset(ybf->u_buffer - 1 - ybf->uv_stride, 127, ybf->uv_width + 5);
  for (i = 0; i < ybf->uv_height; ++i) {
    ybf->u_buffer[ybf->uv_stride * i - 1] = (unsigned char)129;
  }

  memset(ybf->v_buffer - 1 - ybf->uv_stride, 127, ybf->uv_width + 5);
  for (i = 0; i < ybf->uv_height; ++i) {
    ybf->v_buffer[ybf->uv_stride * i - 1] = (unsigned char)129;
  }
}

void vp8_setup_intra_recon_top_line(YV12_BUFFER_CONFIG *ybf) {
  memset(ybf->y_buffer - 1 - ybf->y_stride, 127, ybf->y_width + 5);
  memset(ybf->u_buffer - 1 - ybf->uv_stride, 127, ybf->uv_width + 5);
  memset(ybf->v_buffer - 1 - ybf->uv_stride, 127, ybf->uv_width + 5);
}
