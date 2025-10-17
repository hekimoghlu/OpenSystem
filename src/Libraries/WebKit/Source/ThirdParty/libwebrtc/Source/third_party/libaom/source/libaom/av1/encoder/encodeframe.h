/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#ifndef AOM_AV1_ENCODER_ENCODEFRAME_H_
#define AOM_AV1_ENCODER_ENCODEFRAME_H_

#include "aom/aom_integer.h"
#include "av1/common/blockd.h"
#include "av1/common/enums.h"

#include "av1/encoder/global_motion.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DELTA_Q_PERCEPTUAL_MODULATION \
  1  // 0: variance based
     // 1: wavelet AC energy based

struct macroblock;
struct yv12_buffer_config;
struct AV1_COMP;
struct ThreadData;

void av1_init_rtc_counters(struct macroblock *const x);

void av1_accumulate_rtc_counters(struct AV1_COMP *cpi,
                                 const struct macroblock *const x);

void av1_setup_src_planes(struct macroblock *x,
                          const struct yv12_buffer_config *src, int mi_row,
                          int mi_col, const int num_planes, BLOCK_SIZE bsize);

void av1_encode_frame(struct AV1_COMP *cpi);

void av1_alloc_tile_data(struct AV1_COMP *cpi);
void av1_init_tile_data(struct AV1_COMP *cpi);
void av1_encode_tile(struct AV1_COMP *cpi, struct ThreadData *td, int tile_row,
                     int tile_col);
void av1_encode_sb_row(struct AV1_COMP *cpi, struct ThreadData *td,
                       int tile_row, int tile_col, int mi_row);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_ENCODEFRAME_H_
