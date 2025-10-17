/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
#ifndef AOM_AOM_DSP_SSIM_H_
#define AOM_AOM_DSP_SSIM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "config/aom_config.h"

#if CONFIG_INTERNAL_STATS
#include "aom_scale/yv12config.h"

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

double aom_get_ssim_metrics(uint8_t *img1, int img1_pitch, uint8_t *img2,
                            int img2_pitch, int width, int height, Ssimv *sv2,
                            Metrics *m, int do_inconsistency);

void aom_lowbd_calc_ssim(const YV12_BUFFER_CONFIG *source,
                         const YV12_BUFFER_CONFIG *dest, double *weight,
                         double *fast_ssim);

double aom_calc_fastssim(const YV12_BUFFER_CONFIG *source,
                         const YV12_BUFFER_CONFIG *dest, double *ssim_y,
                         double *ssim_u, double *ssim_v, uint32_t bd,
                         uint32_t in_bd);

#if CONFIG_AV1_HIGHBITDEPTH
void aom_highbd_calc_ssim(const YV12_BUFFER_CONFIG *source,
                          const YV12_BUFFER_CONFIG *dest, double *weight,
                          uint32_t bd, uint32_t in_bd, double *fast_ssim);
#endif  // CONFIG_AV1_HIGHBITDEPTH

void aom_calc_ssim(const YV12_BUFFER_CONFIG *orig,
                   const YV12_BUFFER_CONFIG *recon, const uint32_t bit_depth,
                   const uint32_t in_bit_depth, int is_hbd, double *weight,
                   double *frame_ssim2);
#endif  // CONFIG_INTERNAL_STATS

double aom_ssim2(const uint8_t *img1, const uint8_t *img2, int stride_img1,
                 int stride_img2, int width, int height);

#if CONFIG_AV1_HIGHBITDEPTH
double aom_highbd_ssim2(const uint8_t *img1, const uint8_t *img2,
                        int stride_img1, int stride_img2, int width, int height,
                        uint32_t bd, uint32_t shift);
#endif  // CONFIG_AV1_HIGHBITDEPTH

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_SSIM_H_
