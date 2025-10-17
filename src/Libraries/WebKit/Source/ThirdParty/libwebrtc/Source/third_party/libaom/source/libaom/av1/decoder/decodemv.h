/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#ifndef AOM_AV1_DECODER_DECODEMV_H_
#define AOM_AV1_DECODER_DECODEMV_H_

#include "aom_dsp/bitreader.h"

#include "av1/decoder/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

int av1_neg_deinterleave(int diff, int ref, int max);

void av1_read_mode_info(AV1Decoder *const pbi, DecoderCodingBlock *dcb,
                        aom_reader *r, int x_mis, int y_mis);

#ifdef __cplusplus
}  // extern "C"
#endif

void av1_read_tx_type(const AV1_COMMON *const cm, MACROBLOCKD *xd, int blk_row,
                      int blk_col, TX_SIZE tx_size, aom_reader *r);

#endif  // AOM_AV1_DECODER_DECODEMV_H_
