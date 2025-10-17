/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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
#include "vpx_config.h"
#include "vp8/common/loopfilter.h"

#define prototype_loopfilter(sym)                                      \
  void sym(unsigned char *src, int pitch, const unsigned char *blimit, \
           const unsigned char *limit, const unsigned char *thresh, int count)

#define prototype_loopfilter_nc(sym)                                   \
  void sym(unsigned char *src, int pitch, const unsigned char *blimit, \
           const unsigned char *limit, const unsigned char *thresh)

#define prototype_simple_loopfilter(sym) \
  void sym(unsigned char *y, int ystride, const unsigned char *blimit)

#if HAVE_SSE2 && VPX_ARCH_X86_64
prototype_loopfilter(vp8_loop_filter_bv_y_sse2);
prototype_loopfilter(vp8_loop_filter_bh_y_sse2);
#else
prototype_loopfilter_nc(vp8_loop_filter_vertical_edge_sse2);
prototype_loopfilter_nc(vp8_loop_filter_horizontal_edge_sse2);
#endif
prototype_loopfilter_nc(vp8_mbloop_filter_vertical_edge_sse2);
prototype_loopfilter_nc(vp8_mbloop_filter_horizontal_edge_sse2);

extern loop_filter_uvfunction vp8_loop_filter_horizontal_edge_uv_sse2;
extern loop_filter_uvfunction vp8_loop_filter_vertical_edge_uv_sse2;
extern loop_filter_uvfunction vp8_mbloop_filter_horizontal_edge_uv_sse2;
extern loop_filter_uvfunction vp8_mbloop_filter_vertical_edge_uv_sse2;

/* Horizontal MB filtering */
#if HAVE_SSE2
void vp8_loop_filter_mbh_sse2(unsigned char *y_ptr, unsigned char *u_ptr,
                              unsigned char *v_ptr, int y_stride, int uv_stride,
                              loop_filter_info *lfi) {
  vp8_mbloop_filter_horizontal_edge_sse2(y_ptr, y_stride, lfi->mblim, lfi->lim,
                                         lfi->hev_thr);

  if (u_ptr) {
    vp8_mbloop_filter_horizontal_edge_uv_sse2(u_ptr, uv_stride, lfi->mblim,
                                              lfi->lim, lfi->hev_thr, v_ptr);
  }
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_sse2(unsigned char *y_ptr, unsigned char *u_ptr,
                              unsigned char *v_ptr, int y_stride, int uv_stride,
                              loop_filter_info *lfi) {
  vp8_mbloop_filter_vertical_edge_sse2(y_ptr, y_stride, lfi->mblim, lfi->lim,
                                       lfi->hev_thr);

  if (u_ptr) {
    vp8_mbloop_filter_vertical_edge_uv_sse2(u_ptr, uv_stride, lfi->mblim,
                                            lfi->lim, lfi->hev_thr, v_ptr);
  }
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_sse2(unsigned char *y_ptr, unsigned char *u_ptr,
                             unsigned char *v_ptr, int y_stride, int uv_stride,
                             loop_filter_info *lfi) {
#if VPX_ARCH_X86_64
  vp8_loop_filter_bh_y_sse2(y_ptr, y_stride, lfi->blim, lfi->lim, lfi->hev_thr,
                            2);
#else
  vp8_loop_filter_horizontal_edge_sse2(y_ptr + 4 * y_stride, y_stride,
                                       lfi->blim, lfi->lim, lfi->hev_thr);
  vp8_loop_filter_horizontal_edge_sse2(y_ptr + 8 * y_stride, y_stride,
                                       lfi->blim, lfi->lim, lfi->hev_thr);
  vp8_loop_filter_horizontal_edge_sse2(y_ptr + 12 * y_stride, y_stride,
                                       lfi->blim, lfi->lim, lfi->hev_thr);
#endif

  if (u_ptr) {
    vp8_loop_filter_horizontal_edge_uv_sse2(u_ptr + 4 * uv_stride, uv_stride,
                                            lfi->blim, lfi->lim, lfi->hev_thr,
                                            v_ptr + 4 * uv_stride);
  }
}

void vp8_loop_filter_bhs_sse2(unsigned char *y_ptr, int y_stride,
                              const unsigned char *blimit) {
  vp8_loop_filter_simple_horizontal_edge_sse2(y_ptr + 4 * y_stride, y_stride,
                                              blimit);
  vp8_loop_filter_simple_horizontal_edge_sse2(y_ptr + 8 * y_stride, y_stride,
                                              blimit);
  vp8_loop_filter_simple_horizontal_edge_sse2(y_ptr + 12 * y_stride, y_stride,
                                              blimit);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_sse2(unsigned char *y_ptr, unsigned char *u_ptr,
                             unsigned char *v_ptr, int y_stride, int uv_stride,
                             loop_filter_info *lfi) {
#if VPX_ARCH_X86_64
  vp8_loop_filter_bv_y_sse2(y_ptr, y_stride, lfi->blim, lfi->lim, lfi->hev_thr,
                            2);
#else
  vp8_loop_filter_vertical_edge_sse2(y_ptr + 4, y_stride, lfi->blim, lfi->lim,
                                     lfi->hev_thr);
  vp8_loop_filter_vertical_edge_sse2(y_ptr + 8, y_stride, lfi->blim, lfi->lim,
                                     lfi->hev_thr);
  vp8_loop_filter_vertical_edge_sse2(y_ptr + 12, y_stride, lfi->blim, lfi->lim,
                                     lfi->hev_thr);
#endif

  if (u_ptr) {
    vp8_loop_filter_vertical_edge_uv_sse2(u_ptr + 4, uv_stride, lfi->blim,
                                          lfi->lim, lfi->hev_thr, v_ptr + 4);
  }
}

void vp8_loop_filter_bvs_sse2(unsigned char *y_ptr, int y_stride,
                              const unsigned char *blimit) {
  vp8_loop_filter_simple_vertical_edge_sse2(y_ptr + 4, y_stride, blimit);
  vp8_loop_filter_simple_vertical_edge_sse2(y_ptr + 8, y_stride, blimit);
  vp8_loop_filter_simple_vertical_edge_sse2(y_ptr + 12, y_stride, blimit);
}

#endif
