/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
#ifndef AOM_AV1_ENCODER_AV1_FWD_TXFM1D_CFG_H_
#define AOM_AV1_ENCODER_AV1_FWD_TXFM1D_CFG_H_
#include "av1/common/enums.h"
#include "av1/encoder/av1_fwd_txfm1d.h"
extern const int8_t *av1_fwd_txfm_shift_ls[TX_SIZES_ALL];
extern const int8_t av1_fwd_cos_bit_col[5][5];
extern const int8_t av1_fwd_cos_bit_row[5][5];
#endif  // AOM_AV1_ENCODER_AV1_FWD_TXFM1D_CFG_H_
