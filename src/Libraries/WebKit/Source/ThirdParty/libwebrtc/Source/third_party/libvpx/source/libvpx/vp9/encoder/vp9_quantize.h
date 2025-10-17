/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#ifndef VPX_VP9_ENCODER_VP9_QUANTIZE_H_
#define VPX_VP9_ENCODER_VP9_QUANTIZE_H_

#include "./vpx_config.h"
#include "vp9/encoder/vp9_block.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  DECLARE_ALIGNED(16, int16_t, y_quant[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, y_quant_shift[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, y_zbin[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, y_round[QINDEX_RANGE][8]);

  // TODO(jingning): in progress of re-working the quantization. will decide
  // if we want to deprecate the current use of y_quant.
  DECLARE_ALIGNED(16, int16_t, y_quant_fp[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, uv_quant_fp[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, y_round_fp[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, uv_round_fp[QINDEX_RANGE][8]);

  DECLARE_ALIGNED(16, int16_t, uv_quant[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, uv_quant_shift[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, uv_zbin[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, uv_round[QINDEX_RANGE][8]);
} QUANTS;

struct VP9_COMP;
struct VP9Common;

void vp9_frame_init_quantizer(struct VP9_COMP *cpi);

void vp9_init_plane_quantizers(struct VP9_COMP *cpi, MACROBLOCK *x);

void vp9_init_quantizer(struct VP9_COMP *cpi);

void vp9_set_quantizer(struct VP9_COMP *cm, int q);

int vp9_quantizer_to_qindex(int quantizer);

int vp9_qindex_to_quantizer(int qindex);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_QUANTIZE_H_
