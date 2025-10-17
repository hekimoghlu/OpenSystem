/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#ifndef AOM_AV1_COMMON_AV1_INV_TXFM1D_CFG_H_
#define AOM_AV1_COMMON_AV1_INV_TXFM1D_CFG_H_
#include "av1/common/av1_inv_txfm1d.h"

// sum of fwd_shift_##
static const int8_t inv_start_range[TX_SIZES_ALL] = {
  5,  // 4x4 transform
  6,  // 8x8 transform
  7,  // 16x16 transform
  7,  // 32x32 transform
  7,  // 64x64 transform
  5,  // 4x8 transform
  5,  // 8x4 transform
  6,  // 8x16 transform
  6,  // 16x8 transform
  6,  // 16x32 transform
  6,  // 32x16 transform
  6,  // 32x64 transform
  6,  // 64x32 transform
  6,  // 4x16 transform
  6,  // 16x4 transform
  7,  // 8x32 transform
  7,  // 32x8 transform
  7,  // 16x64 transform
  7,  // 64x16 transform
};

extern const int8_t *av1_inv_txfm_shift_ls[TX_SIZES_ALL];

// Values in both av1_inv_cos_bit_col and av1_inv_cos_bit_row are always 12
// for each valid row and col combination
#define INV_COS_BIT 12

#endif  // AOM_AV1_COMMON_AV1_INV_TXFM1D_CFG_H_
