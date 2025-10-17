/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
#ifndef AOM_AOM_DSP_PSNR_H_
#define AOM_AOM_DSP_PSNR_H_

#include "aom_scale/yv12config.h"
#include "config/aom_config.h"

#define MAX_PSNR 100.0

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double psnr[4];           // total/y/u/v
  uint64_t sse[4];          // total/y/u/v
  uint32_t samples[4];      // total/y/u/v
  double psnr_hbd[4];       // total/y/u/v when input-bit-depth < bit-depth
  uint64_t sse_hbd[4];      // total/y/u/v when input-bit-depth < bit-depth
  uint32_t samples_hbd[4];  // total/y/u/v when input-bit-depth < bit-depth
} PSNR_STATS;

#if CONFIG_INTERNAL_STATS
/*!\brief Converts SSE to PSNR
 *
 * Converts sum of squared errros (SSE) to peak signal-to-noise ratio (PSNR).
 *
 * \param[in]    samples       Number of samples
 * \param[in]    peak          Max sample value
 * \param[in]    sse           Sum of squared errors
 */
double aom_sse_to_psnr(double samples, double peak, double sse);
#endif  // CONFIG_INTERNAL_STATS
uint64_t aom_get_y_var(const YV12_BUFFER_CONFIG *a, int hstart, int width,
                       int vstart, int height);
uint64_t aom_get_u_var(const YV12_BUFFER_CONFIG *a, int hstart, int width,
                       int vstart, int height);
uint64_t aom_get_v_var(const YV12_BUFFER_CONFIG *a, int hstart, int width,
                       int vstart, int height);
int64_t aom_get_y_sse_part(const YV12_BUFFER_CONFIG *a,
                           const YV12_BUFFER_CONFIG *b, int hstart, int width,
                           int vstart, int height);
int64_t aom_get_y_sse(const YV12_BUFFER_CONFIG *a, const YV12_BUFFER_CONFIG *b);
int64_t aom_get_u_sse_part(const YV12_BUFFER_CONFIG *a,
                           const YV12_BUFFER_CONFIG *b, int hstart, int width,
                           int vstart, int height);
int64_t aom_get_u_sse(const YV12_BUFFER_CONFIG *a, const YV12_BUFFER_CONFIG *b);
int64_t aom_get_v_sse_part(const YV12_BUFFER_CONFIG *a,
                           const YV12_BUFFER_CONFIG *b, int hstart, int width,
                           int vstart, int height);
int64_t aom_get_v_sse(const YV12_BUFFER_CONFIG *a, const YV12_BUFFER_CONFIG *b);
int64_t aom_get_sse_plane(const YV12_BUFFER_CONFIG *a,
                          const YV12_BUFFER_CONFIG *b, int plane, int highbd);
#if CONFIG_AV1_HIGHBITDEPTH
uint64_t aom_highbd_get_y_var(const YV12_BUFFER_CONFIG *a, int hstart,
                              int width, int vstart, int height);
uint64_t aom_highbd_get_u_var(const YV12_BUFFER_CONFIG *a, int hstart,
                              int width, int vstart, int height);
uint64_t aom_highbd_get_v_var(const YV12_BUFFER_CONFIG *a, int hstart,
                              int width, int vstart, int height);
int64_t aom_highbd_get_y_sse_part(const YV12_BUFFER_CONFIG *a,
                                  const YV12_BUFFER_CONFIG *b, int hstart,
                                  int width, int vstart, int height);
int64_t aom_highbd_get_y_sse(const YV12_BUFFER_CONFIG *a,
                             const YV12_BUFFER_CONFIG *b);
int64_t aom_highbd_get_u_sse_part(const YV12_BUFFER_CONFIG *a,
                                  const YV12_BUFFER_CONFIG *b, int hstart,
                                  int width, int vstart, int height);
int64_t aom_highbd_get_u_sse(const YV12_BUFFER_CONFIG *a,
                             const YV12_BUFFER_CONFIG *b);
int64_t aom_highbd_get_v_sse_part(const YV12_BUFFER_CONFIG *a,
                                  const YV12_BUFFER_CONFIG *b, int hstart,
                                  int width, int vstart, int height);
int64_t aom_highbd_get_v_sse(const YV12_BUFFER_CONFIG *a,
                             const YV12_BUFFER_CONFIG *b);
void aom_calc_highbd_psnr(const YV12_BUFFER_CONFIG *a,
                          const YV12_BUFFER_CONFIG *b, PSNR_STATS *psnr,
                          unsigned int bit_depth, unsigned int in_bit_depth);
#endif
void aom_calc_psnr(const YV12_BUFFER_CONFIG *a, const YV12_BUFFER_CONFIG *b,
                   PSNR_STATS *psnr);

double aom_psnrhvs(const YV12_BUFFER_CONFIG *source,
                   const YV12_BUFFER_CONFIG *dest, double *phvs_y,
                   double *phvs_u, double *phvs_v, uint32_t bd, uint32_t in_bd);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AOM_DSP_PSNR_H_
