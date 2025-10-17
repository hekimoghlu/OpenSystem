/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#ifndef VPX_VP9_VP9_IFACE_COMMON_H_
#define VPX_VP9_VP9_IFACE_COMMON_H_

#include <assert.h>
#include "vpx_ports/mem.h"
#include "vpx/vp8.h"
#include "vpx_scale/yv12config.h"
#include "common/vp9_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

void yuvconfig2image(vpx_image_t *img, const YV12_BUFFER_CONFIG *yv12,
                     void *user_priv);

vpx_codec_err_t image2yuvconfig(const vpx_image_t *img,
                                YV12_BUFFER_CONFIG *yv12);

static INLINE VP9_REFFRAME
ref_frame_to_vp9_reframe(vpx_ref_frame_type_t frame) {
  switch (frame) {
    case VP8_LAST_FRAME: return VP9_LAST_FLAG;
    case VP8_GOLD_FRAME: return VP9_GOLD_FLAG;
    case VP8_ALTR_FRAME: return VP9_ALT_FLAG;
  }
  assert(0 && "Invalid Reference Frame");
  return VP9_LAST_FLAG;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_VP9_IFACE_COMMON_H_
