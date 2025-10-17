/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#ifndef VPX_VP8_COMMON_LOOPFILTER_H_
#define VPX_VP8_COMMON_LOOPFILTER_H_

#include "vpx_ports/mem.h"
#include "vpx_config.h"
#include "vp8_rtcd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_LOOP_FILTER 63
/* fraction of total macroblock rows to be used in fast filter level picking */
/* has to be > 2 */
#define PARTIAL_FRAME_FRACTION 8

typedef enum { NORMAL_LOOPFILTER = 0, SIMPLE_LOOPFILTER = 1 } LOOPFILTERTYPE;

#if VPX_ARCH_ARM
#define SIMD_WIDTH 1
#else
#define SIMD_WIDTH 16
#endif

/* Need to align this structure so when it is declared and
 * passed it can be loaded into vector registers.
 */
typedef struct {
  DECLARE_ALIGNED(SIMD_WIDTH, unsigned char,
                  mblim[MAX_LOOP_FILTER + 1][SIMD_WIDTH]);
  DECLARE_ALIGNED(SIMD_WIDTH, unsigned char,
                  blim[MAX_LOOP_FILTER + 1][SIMD_WIDTH]);
  DECLARE_ALIGNED(SIMD_WIDTH, unsigned char,
                  lim[MAX_LOOP_FILTER + 1][SIMD_WIDTH]);
  DECLARE_ALIGNED(SIMD_WIDTH, unsigned char, hev_thr[4][SIMD_WIDTH]);
  unsigned char lvl[4][4][4];
  unsigned char hev_thr_lut[2][MAX_LOOP_FILTER + 1];
  unsigned char mode_lf_lut[10];
} loop_filter_info_n;

typedef struct loop_filter_info {
  const unsigned char *mblim;
  const unsigned char *blim;
  const unsigned char *lim;
  const unsigned char *hev_thr;
} loop_filter_info;

typedef void loop_filter_uvfunction(unsigned char *u, /* source pointer */
                                    int p,            /* pitch */
                                    const unsigned char *blimit,
                                    const unsigned char *limit,
                                    const unsigned char *thresh,
                                    unsigned char *v);

/* assorted loopfilter functions which get used elsewhere */
struct VP8Common;
struct macroblockd;
struct modeinfo;

void vp8_loop_filter_init(struct VP8Common *cm);

void vp8_loop_filter_frame_init(struct VP8Common *cm, struct macroblockd *mbd,
                                int default_filt_lvl);

void vp8_loop_filter_frame(struct VP8Common *cm, struct macroblockd *mbd,
                           int frame_type);

void vp8_loop_filter_partial_frame(struct VP8Common *cm,
                                   struct macroblockd *mbd,
                                   int default_filt_lvl);

void vp8_loop_filter_frame_yonly(struct VP8Common *cm, struct macroblockd *mbd,
                                 int default_filt_lvl);

void vp8_loop_filter_update_sharpness(loop_filter_info_n *lfi,
                                      int sharpness_lvl);

void vp8_loop_filter_row_normal(struct VP8Common *cm,
                                struct modeinfo *mode_info_context, int mb_row,
                                int post_ystride, int post_uvstride,
                                unsigned char *y_ptr, unsigned char *u_ptr,
                                unsigned char *v_ptr);

void vp8_loop_filter_row_simple(struct VP8Common *cm,
                                struct modeinfo *mode_info_context, int mb_row,
                                int post_ystride, unsigned char *y_ptr);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_LOOPFILTER_H_
