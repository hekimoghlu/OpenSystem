/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
#ifndef VPX_VP8_COMMON_ONYXD_H_
#define VPX_VP8_COMMON_ONYXD_H_

/* Create/destroy static data structures. */
#ifdef __cplusplus
extern "C" {
#endif
#include "vpx_scale/yv12config.h"
#include "ppflags.h"
#include "vpx_ports/mem.h"
#include "vpx/vpx_codec.h"
#include "vpx/vp8.h"

struct VP8D_COMP;
struct VP8Common;

typedef struct {
  int Width;
  int Height;
  int Version;
  int postprocess;
  int max_threads;
  int error_concealment;
} VP8D_CONFIG;

typedef enum { VP8D_OK = 0 } VP8D_SETTING;

void vp8dx_initialize(void);

void vp8dx_set_setting(struct VP8D_COMP *comp, VP8D_SETTING oxst, int x);

int vp8dx_get_setting(struct VP8D_COMP *comp, VP8D_SETTING oxst);

int vp8dx_receive_compressed_data(struct VP8D_COMP *pbi);
int vp8dx_get_raw_frame(struct VP8D_COMP *pbi, YV12_BUFFER_CONFIG *sd,
                        vp8_ppflags_t *flags);
int vp8dx_references_buffer(struct VP8Common *oci, int ref_frame);

vpx_codec_err_t vp8dx_get_reference(struct VP8D_COMP *pbi,
                                    enum vpx_ref_frame_type ref_frame_flag,
                                    YV12_BUFFER_CONFIG *sd);
vpx_codec_err_t vp8dx_set_reference(struct VP8D_COMP *pbi,
                                    enum vpx_ref_frame_type ref_frame_flag,
                                    YV12_BUFFER_CONFIG *sd);
int vp8dx_get_quantizer(const struct VP8D_COMP *pbi);

#ifdef __cplusplus
}
#endif

#endif  // VPX_VP8_COMMON_ONYXD_H_
