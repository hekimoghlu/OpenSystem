/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
/**
 * SvcContext - input parameters and state to encode a multi-layered
 * spatial SVC frame
 */

#ifndef VPX_EXAMPLES_SVC_CONTEXT_H_
#define VPX_EXAMPLES_SVC_CONTEXT_H_

#include "vpx/vp8cx.h"
#include "vpx/vpx_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SVC_LOG_LEVEL {
  SVC_LOG_ERROR,
  SVC_LOG_INFO,
  SVC_LOG_DEBUG
} SVC_LOG_LEVEL;

typedef struct {
  // public interface to svc_command options
  int spatial_layers;   // number of spatial layers
  int temporal_layers;  // number of temporal layers
  int temporal_layering_mode;
  SVC_LOG_LEVEL log_level;  // amount of information to display
  int output_rc_stat;       // for outputting rc stats
  int speed;                // speed setting for codec
  int threads;
  int aqmode;  // turns on aq-mode=3 (cyclic_refresh): 0=off, 1=on.
  // private storage for vpx_svc_encode
  void *internal;
} SvcContext;

#define OPTION_BUFFER_SIZE 1024
#define COMPONENTS 4  // psnr & sse statistics maintained for total, y, u, v

typedef struct SvcInternal {
  char options[OPTION_BUFFER_SIZE];  // set by vpx_svc_set_options

  // values extracted from option, quantizers
  vpx_svc_extra_cfg_t svc_params;
  int enable_auto_alt_ref[VPX_SS_MAX_LAYERS];
  int bitrates[VPX_MAX_LAYERS];

  // accumulated statistics
  double psnr_sum[VPX_SS_MAX_LAYERS][COMPONENTS];  // total/Y/U/V
  uint64_t sse_sum[VPX_SS_MAX_LAYERS][COMPONENTS];
  uint32_t bytes_sum[VPX_SS_MAX_LAYERS];

  // codec encoding values
  int width;    // width of highest layer
  int height;   // height of highest layer
  int kf_dist;  // distance between keyframes

  // state variables
  int psnr_pkt_received;
  int layer;
  int use_multiple_frame_contexts;

  vpx_codec_ctx_t *codec_ctx;
} SvcInternal_t;

/**
 * Set SVC options
 * options are supplied as a single string separated by spaces
 * Format: encoding-mode=<i|ip|alt-ip|gf>
 *         layers=<layer_count>
 *         scaling-factors=<n1>/<d1>,<n2>/<d2>,...
 *         quantizers=<q1>,<q2>,...
 */
vpx_codec_err_t vpx_svc_set_options(SvcContext *svc_ctx, const char *options);

/**
 * initialize SVC encoding
 */
vpx_codec_err_t vpx_svc_init(SvcContext *svc_ctx, vpx_codec_ctx_t *codec_ctx,
                             vpx_codec_iface_t *iface,
                             vpx_codec_enc_cfg_t *cfg);
/**
 * encode a frame of video with multiple layers
 */
vpx_codec_err_t vpx_svc_encode(SvcContext *svc_ctx, vpx_codec_ctx_t *codec_ctx,
                               struct vpx_image *rawimg, vpx_codec_pts_t pts,
                               int64_t duration, int deadline);

/**
 * finished with svc encoding, release allocated resources
 */
void vpx_svc_release(SvcContext *svc_ctx);

/**
 * dump accumulated statistics and reset accumulated values
 */
void vpx_svc_dump_statistics(SvcContext *svc_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_EXAMPLES_SVC_CONTEXT_H_
