/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#ifndef AOM_AV1_ENCODER_AQ_VARIANCE_H_
#define AOM_AV1_ENCODER_AQ_VARIANCE_H_

#include "av1/encoder/encoder.h"
#include "config/aom_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !CONFIG_REALTIME_ONLY
void av1_vaq_frame_setup(AV1_COMP *cpi);

int av1_log_block_avg(const AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bs,
                      int mi_row, int mi_col);
int av1_compute_q_from_energy_level_deltaq_mode(const AV1_COMP *const cpi,
                                                int block_var_level);
int av1_block_wavelet_energy_level(const AV1_COMP *cpi, MACROBLOCK *x,
                                   BLOCK_SIZE bs);
#endif  // !CONFIG_REALTIME_ONLY

int av1_log_block_var(const AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_AQ_VARIANCE_H_
