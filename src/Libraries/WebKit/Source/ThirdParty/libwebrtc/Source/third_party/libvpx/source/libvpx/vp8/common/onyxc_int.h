/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#ifndef VPX_VP8_COMMON_ONYXC_INT_H_
#define VPX_VP8_COMMON_ONYXC_INT_H_

#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx/internal/vpx_codec_internal.h"
#include "loopfilter.h"
#include "entropymv.h"
#include "entropy.h"
#if CONFIG_POSTPROC
#include "postproc.h"
#endif

/*#ifdef PACKET_TESTING*/
#include "header.h"
/*#endif*/

#ifdef __cplusplus
extern "C" {
#endif

#define MINQ 0
#define MAXQ 127
#define QINDEX_RANGE (MAXQ + 1)

#define NUM_YV12_BUFFERS 4

#define MAX_PARTITIONS 9

typedef struct frame_contexts {
  vp8_prob bmode_prob[VP8_BINTRAMODES - 1];
  vp8_prob ymode_prob[VP8_YMODES - 1]; /* interframe intra mode probs */
  vp8_prob uv_mode_prob[VP8_UV_MODES - 1];
  vp8_prob sub_mv_ref_prob[VP8_SUBMVREFS - 1];
  vp8_prob coef_probs[BLOCK_TYPES][COEF_BANDS][PREV_COEF_CONTEXTS]
                     [ENTROPY_NODES];
  MV_CONTEXT mvc[2];
} FRAME_CONTEXT;

typedef enum {
  ONE_PARTITION = 0,
  TWO_PARTITION = 1,
  FOUR_PARTITION = 2,
  EIGHT_PARTITION = 3
} TOKEN_PARTITION;

typedef enum {
  RECON_CLAMP_REQUIRED = 0,
  RECON_CLAMP_NOTREQUIRED = 1
} CLAMP_TYPE;

typedef struct VP8Common {
  struct vpx_internal_error_info error;

  DECLARE_ALIGNED(16, short, Y1dequant[QINDEX_RANGE][2]);
  DECLARE_ALIGNED(16, short, Y2dequant[QINDEX_RANGE][2]);
  DECLARE_ALIGNED(16, short, UVdequant[QINDEX_RANGE][2]);

  int Width;
  int Height;
  int horiz_scale;
  int vert_scale;

  CLAMP_TYPE clamp_type;

  YV12_BUFFER_CONFIG *frame_to_show;

  YV12_BUFFER_CONFIG yv12_fb[NUM_YV12_BUFFERS];
  int fb_idx_ref_cnt[NUM_YV12_BUFFERS];
  int new_fb_idx, lst_fb_idx, gld_fb_idx, alt_fb_idx;

  YV12_BUFFER_CONFIG temp_scale_frame;

#if CONFIG_POSTPROC
  YV12_BUFFER_CONFIG post_proc_buffer;
  YV12_BUFFER_CONFIG post_proc_buffer_int;
  int post_proc_buffer_int_used;
  unsigned char *pp_limits_buffer; /* post-processing filter coefficients */
#endif

  FRAME_TYPE
  last_frame_type; /* Save last frame's frame type for motion search. */
  FRAME_TYPE frame_type;

  int show_frame;

  int frame_flags;
  int MBs;
  int mb_rows;
  int mb_cols;
  int mode_info_stride;

  /* profile settings */
  int mb_no_coeff_skip;
  int no_lpf;
  int use_bilinear_mc_filter;
  int full_pixel;

  int base_qindex;

  int y1dc_delta_q;
  int y2dc_delta_q;
  int y2ac_delta_q;
  int uvdc_delta_q;
  int uvac_delta_q;

  /* We allocate a MODE_INFO struct for each macroblock, together with
     an extra row on top and column on the left to simplify prediction. */

  MODE_INFO *mip; /* Base of allocated array */
  MODE_INFO *mi;  /* Corresponds to upper left visible macroblock */
#if CONFIG_ERROR_CONCEALMENT
  MODE_INFO *prev_mip; /* MODE_INFO array 'mip' from last decoded frame */
  MODE_INFO *prev_mi;  /* 'mi' from last frame (points into prev_mip) */
#endif
  /* MODE_INFO for the last decoded frame to show */
  MODE_INFO *show_frame_mi;
  LOOPFILTERTYPE filter_type;

  loop_filter_info_n lf_info;

  int filter_level;
  int last_sharpness_level;
  int sharpness_level;

  int refresh_last_frame;    /* Two state 0 = NO, 1 = YES */
  int refresh_golden_frame;  /* Two state 0 = NO, 1 = YES */
  int refresh_alt_ref_frame; /* Two state 0 = NO, 1 = YES */

  int copy_buffer_to_gf;  /* 0 none, 1 Last to GF, 2 ARF to GF */
  int copy_buffer_to_arf; /* 0 none, 1 Last to ARF, 2 GF to ARF */

  int refresh_entropy_probs; /* Two state 0 = NO, 1 = YES */

  int ref_frame_sign_bias[MAX_REF_FRAMES]; /* Two state 0, 1 */

  /* Y,U,V,Y2 */
  ENTROPY_CONTEXT_PLANES *above_context; /* row of context for each plane */
  ENTROPY_CONTEXT_PLANES left_context;   /* (up to) 4 contexts "" */

  FRAME_CONTEXT lfc; /* last frame entropy */
  FRAME_CONTEXT fc;  /* this frame entropy */

  unsigned int current_video_frame;

  int version;

  TOKEN_PARTITION multi_token_partition;

#ifdef PACKET_TESTING
  VP8_HEADER oh;
#endif

#if CONFIG_MULTITHREAD
  int processor_core_count;
#endif
#if CONFIG_POSTPROC
  struct postproc_state postproc_state;
#endif
} VP8_COMMON;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_ONYXC_INT_H_
