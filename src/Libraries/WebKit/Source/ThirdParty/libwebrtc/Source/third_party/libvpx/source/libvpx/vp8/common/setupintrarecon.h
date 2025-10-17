/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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
#ifndef VPX_VP8_COMMON_SETUPINTRARECON_H_
#define VPX_VP8_COMMON_SETUPINTRARECON_H_

#include "./vpx_config.h"
#include "vpx_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif
extern void vp8_setup_intra_recon(YV12_BUFFER_CONFIG *ybf);
extern void vp8_setup_intra_recon_top_line(YV12_BUFFER_CONFIG *ybf);

static INLINE void setup_intra_recon_left(unsigned char *y_buffer,
                                          unsigned char *u_buffer,
                                          unsigned char *v_buffer, int y_stride,
                                          int uv_stride) {
  int i;

  for (i = 0; i < 16; ++i) y_buffer[y_stride * i] = (unsigned char)129;

  for (i = 0; i < 8; ++i) u_buffer[uv_stride * i] = (unsigned char)129;

  for (i = 0; i < 8; ++i) v_buffer[uv_stride * i] = (unsigned char)129;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_SETUPINTRARECON_H_
