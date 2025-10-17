/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#ifndef VPX_VP9_COMMON_VP9_POSTPROC_H_
#define VPX_VP9_COMMON_VP9_POSTPROC_H_

#include "vpx_ports/mem.h"
#include "vpx_scale/yv12config.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_mfqe.h"
#include "vp9/common/vp9_ppflags.h"

#ifdef __cplusplus
extern "C" {
#endif

struct postproc_state {
  int last_q;
  int last_noise;
  int last_base_qindex;
  int last_frame_valid;
  MODE_INFO *prev_mip;
  MODE_INFO *prev_mi;
  int clamp;
  uint8_t *limits;
  int8_t *generated_noise;
};

struct VP9Common;

#define MFQE_PRECISION 4

int vp9_post_proc_frame(struct VP9Common *cm, YV12_BUFFER_CONFIG *dest,
                        vp9_ppflags_t *ppflags, int unscaled_width);

void vp9_denoise(struct VP9Common *cm, const YV12_BUFFER_CONFIG *src,
                 YV12_BUFFER_CONFIG *dst, int q, uint8_t *limits);

void vp9_deblock(struct VP9Common *cm, const YV12_BUFFER_CONFIG *src,
                 YV12_BUFFER_CONFIG *dst, int q, uint8_t *limits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_POSTPROC_H_
