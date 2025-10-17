/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
#ifndef VPX_VP9_COMMON_VP9_COMMON_DATA_H_
#define VPX_VP9_COMMON_VP9_COMMON_DATA_H_

#include "vp9/common/vp9_enums.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t b_width_log2_lookup[BLOCK_SIZES];
extern const uint8_t b_height_log2_lookup[BLOCK_SIZES];
extern const uint8_t mi_width_log2_lookup[BLOCK_SIZES];
extern const uint8_t num_8x8_blocks_wide_lookup[BLOCK_SIZES];
extern const uint8_t num_8x8_blocks_high_lookup[BLOCK_SIZES];
extern const uint8_t num_4x4_blocks_high_lookup[BLOCK_SIZES];
extern const uint8_t num_4x4_blocks_wide_lookup[BLOCK_SIZES];
extern const uint8_t size_group_lookup[BLOCK_SIZES];
extern const uint8_t num_pels_log2_lookup[BLOCK_SIZES];
extern const PARTITION_TYPE partition_lookup[][BLOCK_SIZES];
extern const BLOCK_SIZE subsize_lookup[PARTITION_TYPES][BLOCK_SIZES];
extern const TX_SIZE max_txsize_lookup[BLOCK_SIZES];
extern const BLOCK_SIZE txsize_to_bsize[TX_SIZES];
extern const TX_SIZE tx_mode_to_biggest_tx_size[TX_MODES];
extern const BLOCK_SIZE ss_size_lookup[BLOCK_SIZES][2][2];
extern const TX_SIZE uv_txsize_lookup[BLOCK_SIZES][TX_SIZES][2][2];
#if CONFIG_BETTER_HW_COMPATIBILITY && CONFIG_VP9_HIGHBITDEPTH
extern const uint8_t need_top_left[INTRA_MODES];
#endif  // CONFIG_BETTER_HW_COMPATIBILITY && CONFIG_VP9_HIGHBITDEPTH

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_COMMON_DATA_H_
