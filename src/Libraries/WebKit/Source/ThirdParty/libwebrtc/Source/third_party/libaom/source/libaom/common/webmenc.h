/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#ifndef AOM_COMMON_WEBMENC_H_
#define AOM_COMMON_WEBMENC_H_

#include <stdio.h>
#include <stdlib.h>

#include "tools_common.h"
#include "aom/aom_encoder.h"

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
enum {
  STEREO_FORMAT_MONO = 0,
  STEREO_FORMAT_LEFT_RIGHT = 1,
  STEREO_FORMAT_BOTTOM_TOP = 2,
  STEREO_FORMAT_TOP_BOTTOM = 3,
  STEREO_FORMAT_RIGHT_LEFT = 11
} UENUM1BYTE(stereo_format_t);

// Simplistic mechanism to extract encoder settings, without having
// to re-invoke the entire flag-parsing logic. It lists the codec version
// and then copies the arguments as-is from argv, but skips the binary name,
// any arguments that match the input filename, and the output flags "-o"
// and "--output" (and the following argument for those flags). The caller
// is responsible for free-ing the returned string. If there is insufficient
// memory, it returns nullptr.
char *extract_encoder_settings(const char *version, const char **argv, int argc,
                               const char *input_fname);

// The following functions wrap libwebm's mkvmuxer. All functions return 0 upon
// success, or -1 upon failure.

int write_webm_file_header(struct WebmOutputContext *webm_ctx,
                           aom_codec_ctx_t *encoder_ctx,
                           const aom_codec_enc_cfg_t *cfg,
                           stereo_format_t stereo_fmt, unsigned int fourcc,
                           const struct AvxRational *par,
                           const char *encoder_settings);

int write_webm_block(struct WebmOutputContext *webm_ctx,
                     const aom_codec_enc_cfg_t *cfg,
                     const aom_codec_cx_pkt_t *pkt);

int write_webm_file_footer(struct WebmOutputContext *webm_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_WEBMENC_H_
