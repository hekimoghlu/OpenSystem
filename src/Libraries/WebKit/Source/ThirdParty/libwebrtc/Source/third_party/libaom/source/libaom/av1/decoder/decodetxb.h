/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
#ifndef AOM_AV1_DECODER_DECODETXB_H_
#define AOM_AV1_DECODER_DECODETXB_H_

#include "av1/common/enums.h"

struct aom_reader;
struct AV1Common;
struct DecoderCodingBlock;
struct txb_ctx;

void av1_read_coeffs_txb(const struct AV1Common *const cm,
                         struct DecoderCodingBlock *dcb,
                         struct aom_reader *const r, const int plane,
                         const int row, const int col, const TX_SIZE tx_size);
#endif  // AOM_AV1_DECODER_DECODETXB_H_
