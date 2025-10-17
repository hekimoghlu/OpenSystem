/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
#ifndef AOM_AV1_ENCODER_TUNE_BUTTERAUGLI_H_
#define AOM_AV1_ENCODER_TUNE_BUTTERAUGLI_H_

#include "aom_scale/yv12config.h"
#include "av1/common/enums.h"
#include "av1/encoder/ratectrl.h"
#include "av1/encoder/block.h"

typedef struct {
  // Stores the scaling factors for rdmult when tuning for Butteraugli.
  // rdmult_scaling_factors[row * num_cols + col] stores the scaling factors for
  // 4x4 block at (row, col).
  double *rdmult_scaling_factors;
  YV12_BUFFER_CONFIG source, resized_source;
  bool recon_set;
} TuneButteraugliInfo;

struct AV1_COMP;
static const BLOCK_SIZE butteraugli_rdo_bsize = BLOCK_16X16;

void av1_set_butteraugli_rdmult(const struct AV1_COMP *cpi, MACROBLOCK *x,
                                BLOCK_SIZE bsize, int mi_row, int mi_col,
                                int *rdmult);

void av1_setup_butteraugli_source(struct AV1_COMP *cpi);

// 'K' is used to balance the rate-distortion distribution between PSNR
// and Butteraugli.
void av1_setup_butteraugli_rdmult_and_restore_source(struct AV1_COMP *cpi,
                                                     double K);

void av1_setup_butteraugli_rdmult(struct AV1_COMP *cpi);

#endif  // AOM_AV1_ENCODER_TUNE_BUTTERAUGLI_H_
