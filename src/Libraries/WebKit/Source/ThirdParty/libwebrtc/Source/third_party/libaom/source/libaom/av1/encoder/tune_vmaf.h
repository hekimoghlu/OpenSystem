/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#ifndef AOM_AV1_ENCODER_TUNE_VMAF_H_
#define AOM_AV1_ENCODER_TUNE_VMAF_H_

#include "aom_dsp/vmaf.h"
#include "aom_scale/yv12config.h"
#include "av1/common/enums.h"
#include "av1/encoder/ratectrl.h"
#include "av1/encoder/block.h"

typedef struct {
  // Stores the scaling factors for rdmult when tuning for VMAF.
  // rdmult_scaling_factors[row * num_cols + col] stores the scaling factors for
  // 64x64 block at (row, col).
  double *rdmult_scaling_factors;

  // Stores the luma sse of the last frame.
  double last_frame_ysse[MAX_ARF_LAYERS];

  // Stores the VMAF of the last frame.
  double last_frame_vmaf[MAX_ARF_LAYERS];

  // Stores the filter strength of the last frame.
  double last_frame_unsharp_amount[MAX_ARF_LAYERS];

  // Stores the origial qindex before scaling.
  int original_qindex;

  // VMAF model used in VMAF caculations.
  VmafModel *vmaf_model;
} TuneVMAFInfo;

struct AV1_COMP;

void av1_vmaf_blk_preprocessing(struct AV1_COMP *cpi,
                                const YV12_BUFFER_CONFIG *source);

void av1_vmaf_frame_preprocessing(struct AV1_COMP *cpi,
                                  const YV12_BUFFER_CONFIG *source);

void av1_vmaf_neg_preprocessing(struct AV1_COMP *cpi,
                                const YV12_BUFFER_CONFIG *source);

void av1_set_mb_vmaf_rdmult_scaling(struct AV1_COMP *cpi);

void av1_set_vmaf_rdmult(const struct AV1_COMP *cpi, MACROBLOCK *x,
                         BLOCK_SIZE bsize, int mi_row, int mi_col, int *rdmult);

int av1_get_vmaf_base_qindex(const struct AV1_COMP *cpi, int current_qindex);

void av1_update_vmaf_curve(struct AV1_COMP *cpi);

#endif  // AOM_AV1_ENCODER_TUNE_VMAF_H_
