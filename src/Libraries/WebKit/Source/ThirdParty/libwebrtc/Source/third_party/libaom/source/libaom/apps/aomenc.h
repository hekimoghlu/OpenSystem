/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#ifndef AOM_APPS_AOMENC_H_
#define AOM_APPS_AOMENC_H_

#include "aom/aom_codec.h"
#include "aom/aom_encoder.h"
#include "av1/arg_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  I420,  // 4:2:0 8+ bit-depth
  I422,  // 4:2:2 8+ bit-depth
  I444,  // 4:4:4 8+ bit-depth
  YV12,  // 4:2:0 with uv flipped, only 8-bit depth
  NV12,  // 4:2:0 with uv interleaved, only 8-bit depth
} ColorInputType;

/* Configuration elements common to all streams. */
struct AvxEncoderConfig {
  aom_codec_iface_t *codec;
  int passes;
  int pass;
  unsigned int usage;
  ColorInputType color_type;
  int quiet;
  int verbose;
  int limit;
  int skip_frames;
  int show_psnr;
  enum TestDecodeFatality test_decode;
  int have_framerate;
  struct aom_rational framerate;
  int debug;
  int show_q_hist_buckets;
  int show_rate_hist_buckets;
  int disable_warnings;
  int disable_warning_prompt;
  int experimental_bitstream;
  aom_chroma_sample_position_t csp;
  cfg_options_t encoder_config;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_APPS_AOMENC_H_
