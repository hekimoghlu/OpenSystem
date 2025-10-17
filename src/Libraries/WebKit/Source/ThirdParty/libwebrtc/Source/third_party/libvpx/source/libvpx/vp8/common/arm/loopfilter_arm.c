/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#include "./vpx_config.h"
#include "./vp8_rtcd.h"
#include "vp8/common/arm/loopfilter_arm.h"
#include "vp8/common/loopfilter.h"
#include "vp8/common/onyxc_int.h"

/* NEON loopfilter functions */
/* Horizontal MB filtering */
void vp8_loop_filter_mbh_neon(unsigned char *y_ptr, unsigned char *u_ptr,
                              unsigned char *v_ptr, int y_stride, int uv_stride,
                              loop_filter_info *lfi) {
  unsigned char mblim = *lfi->mblim;
  unsigned char lim = *lfi->lim;
  unsigned char hev_thr = *lfi->hev_thr;
  vp8_mbloop_filter_horizontal_edge_y_neon(y_ptr, y_stride, mblim, lim,
                                           hev_thr);

  if (u_ptr)
    vp8_mbloop_filter_horizontal_edge_uv_neon(u_ptr, uv_stride, mblim, lim,
                                              hev_thr, v_ptr);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_neon(unsigned char *y_ptr, unsigned char *u_ptr,
                              unsigned char *v_ptr, int y_stride, int uv_stride,
                              loop_filter_info *lfi) {
  unsigned char mblim = *lfi->mblim;
  unsigned char lim = *lfi->lim;
  unsigned char hev_thr = *lfi->hev_thr;

  vp8_mbloop_filter_vertical_edge_y_neon(y_ptr, y_stride, mblim, lim, hev_thr);

  if (u_ptr)
    vp8_mbloop_filter_vertical_edge_uv_neon(u_ptr, uv_stride, mblim, lim,
                                            hev_thr, v_ptr);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_neon(unsigned char *y_ptr, unsigned char *u_ptr,
                             unsigned char *v_ptr, int y_stride, int uv_stride,
                             loop_filter_info *lfi) {
  unsigned char blim = *lfi->blim;
  unsigned char lim = *lfi->lim;
  unsigned char hev_thr = *lfi->hev_thr;

  vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 4 * y_stride, y_stride, blim,
                                         lim, hev_thr);
  vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 8 * y_stride, y_stride, blim,
                                         lim, hev_thr);
  vp8_loop_filter_horizontal_edge_y_neon(y_ptr + 12 * y_stride, y_stride, blim,
                                         lim, hev_thr);

  if (u_ptr)
    vp8_loop_filter_horizontal_edge_uv_neon(u_ptr + 4 * uv_stride, uv_stride,
                                            blim, lim, hev_thr,
                                            v_ptr + 4 * uv_stride);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_neon(unsigned char *y_ptr, unsigned char *u_ptr,
                             unsigned char *v_ptr, int y_stride, int uv_stride,
                             loop_filter_info *lfi) {
  unsigned char blim = *lfi->blim;
  unsigned char lim = *lfi->lim;
  unsigned char hev_thr = *lfi->hev_thr;

  vp8_loop_filter_vertical_edge_y_neon(y_ptr + 4, y_stride, blim, lim, hev_thr);
  vp8_loop_filter_vertical_edge_y_neon(y_ptr + 8, y_stride, blim, lim, hev_thr);
  vp8_loop_filter_vertical_edge_y_neon(y_ptr + 12, y_stride, blim, lim,
                                       hev_thr);

  if (u_ptr)
    vp8_loop_filter_vertical_edge_uv_neon(u_ptr + 4, uv_stride, blim, lim,
                                          hev_thr, v_ptr + 4);
}
