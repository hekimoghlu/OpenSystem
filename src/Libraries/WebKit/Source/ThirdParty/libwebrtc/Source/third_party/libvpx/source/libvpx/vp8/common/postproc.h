/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#ifndef VPX_VP8_COMMON_POSTPROC_H_
#define VPX_VP8_COMMON_POSTPROC_H_

#include "vpx_ports/mem.h"
struct postproc_state {
  int last_q;
  int last_noise;
  int last_base_qindex;
  int last_frame_valid;
  int clamp;
  int8_t *generated_noise;
};
#include "onyxc_int.h"
#include "ppflags.h"

#ifdef __cplusplus
extern "C" {
#endif
int vp8_post_proc_frame(struct VP8Common *oci, YV12_BUFFER_CONFIG *dest,
                        vp8_ppflags_t *ppflags);

void vp8_de_noise(struct VP8Common *cm, YV12_BUFFER_CONFIG *source, int q,
                  int uvfilter);

void vp8_deblock(struct VP8Common *cm, YV12_BUFFER_CONFIG *source,
                 YV12_BUFFER_CONFIG *post, int q);

#define MFQE_PRECISION 4

void vp8_multiframe_quality_enhance(struct VP8Common *cm);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_POSTPROC_H_
