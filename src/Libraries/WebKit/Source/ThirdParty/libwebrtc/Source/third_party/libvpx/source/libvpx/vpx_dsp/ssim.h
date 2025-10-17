/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#ifndef VPX_VPX_DSP_SSIM_H_
#define VPX_VPX_DSP_SSIM_H_

#define MAX_SSIM_DB 100.0;

#ifdef __cplusplus
extern "C" {
#endif

#include "./vpx_config.h"
#include "vpx_scale/yv12config.h"

// metrics used for calculating ssim, ssim2, dssim, and ssimc
typedef struct {
  // source sum ( over 8x8 region )
  uint32_t sum_s;

  // reference sum (over 8x8 region )
  uint32_t sum_r;

  // source sum squared ( over 8x8 region )
  uint32_t sum_sq_s;

  // reference sum squared (over 8x8 region )
  uint32_t sum_sq_r;

  // sum of source times reference (over 8x8 region)
  uint32_t sum_sxr;

  // calculated ssim score between source and reference
  double ssim;
} Ssimv;

// metrics collected on a frame basis
typedef struct {
  // ssim consistency error metric ( see code for explanation )
  double ssimc;

  // standard ssim
  double ssim;

  // revised ssim ( see code for explanation)
  double ssim2;

  // ssim restated as an error metric like sse
  double dssim;

  // dssim converted to decibels
  double dssimd;

  // ssimc converted to decibels
  double ssimcd;
} Metrics;

double vpx_get_ssim_metrics(uint8_t *img1, int img1_pitch, uint8_t *img2,
                            int img2_pitch, int width, int height, Ssimv *sv2,
                            Metrics *m, int do_inconsistency);

double vpx_calc_ssim(const YV12_BUFFER_CONFIG *source,
                     const YV12_BUFFER_CONFIG *dest, double *weight);

double vpx_calc_fastssim(const YV12_BUFFER_CONFIG *source,
                         const YV12_BUFFER_CONFIG *dest, double *ssim_y,
                         double *ssim_u, double *ssim_v, uint32_t bd,
                         uint32_t in_bd);

#if CONFIG_VP9_HIGHBITDEPTH
double vpx_highbd_calc_ssim(const YV12_BUFFER_CONFIG *source,
                            const YV12_BUFFER_CONFIG *dest, double *weight,
                            uint32_t bd, uint32_t in_bd);
#endif  // CONFIG_VP9_HIGHBITDEPTH

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_SSIM_H_
