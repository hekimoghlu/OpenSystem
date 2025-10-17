/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#ifndef VPX_WEBMENC_H_
#define VPX_WEBMENC_H_

#include <stdio.h>
#include <stdlib.h>

#include "tools_common.h"
#include "vpx/vpx_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

struct WebmOutputContext {
  int debug;
  FILE *stream;
  int64_t last_pts_ns;
  void *writer;
  void *segment;
};

/* Stereo 3D packed frame format */
typedef enum stereo_format {
  STEREO_FORMAT_MONO = 0,
  STEREO_FORMAT_LEFT_RIGHT = 1,
  STEREO_FORMAT_BOTTOM_TOP = 2,
  STEREO_FORMAT_TOP_BOTTOM = 3,
  STEREO_FORMAT_RIGHT_LEFT = 11
} stereo_format_t;

void write_webm_file_header(struct WebmOutputContext *webm_ctx,
                            const vpx_codec_enc_cfg_t *cfg,
                            stereo_format_t stereo_fmt, unsigned int fourcc,
                            const struct VpxRational *par);

void write_webm_block(struct WebmOutputContext *webm_ctx,
                      const vpx_codec_enc_cfg_t *cfg,
                      const vpx_codec_cx_pkt_t *pkt);

void write_webm_file_footer(struct WebmOutputContext *webm_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_WEBMENC_H_
