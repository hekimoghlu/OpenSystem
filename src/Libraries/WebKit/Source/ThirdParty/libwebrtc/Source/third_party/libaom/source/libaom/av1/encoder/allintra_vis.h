/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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
#ifndef AOM_AV1_ENCODER_ALLINTRA_VIS_H_
#define AOM_AV1_ENCODER_ALLINTRA_VIS_H_

#include "config/aom_dsp_rtcd.h"

#include "av1/common/enums.h"
#include "av1/common/reconintra.h"

#include "av1/encoder/block.h"
#include "av1/encoder/encoder.h"

#define MB_WIENER_MT_UNIT_SIZE BLOCK_64X64

void av1_init_mb_wiener_var_buffer(AV1_COMP *cpi);

void av1_calc_mb_wiener_var_row(AV1_COMP *const cpi, MACROBLOCK *x,
                                MACROBLOCKD *xd, const int mi_row,
                                int16_t *src_diff, tran_low_t *coeff,
                                tran_low_t *qcoeff, tran_low_t *dqcoeff,
                                double *sum_rec_distortion,
                                double *sum_est_rate, uint8_t *pred_buffer);

void av1_set_mb_wiener_variance(AV1_COMP *cpi);

int av1_get_sbq_perceptual_ai(AV1_COMP *const cpi, BLOCK_SIZE bsize, int mi_row,
                              int mi_col);

// User rating based mode
void av1_init_mb_ur_var_buffer(AV1_COMP *cpi);

void av1_set_mb_ur_variance(AV1_COMP *cpi);

int av1_get_sbq_user_rating_based(AV1_COMP *const cpi, int mi_row, int mi_col);

#endif  // AOM_AV1_ENCODER_ALLINTRA_VIS_H_
