/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
#ifndef VPX_VP9_ENCODER_VP9_TEMPORAL_FILTER_H_
#define VPX_VP9_ENCODER_VP9_TEMPORAL_FILTER_H_

#ifdef __cplusplus
extern "C" {
#endif

#define ARNR_FILT_QINDEX 128
struct VP9_COMP;
struct ThreadData;

// Block size used in temporal filtering
#define TF_BLOCK BLOCK_32X32
#define BH 32
#define BH_LOG2 5
#define BW 32
#define BW_LOG2 5
#define BLK_PELS ((BH) * (BW))  // Pixels in the block
#define TF_SHIFT 2
#define TF_ROUND 3
#define THR_SHIFT 2
#define TF_SUB_BLOCK BLOCK_16X16
#define SUB_BH 16
#define SUB_BW 16
#define MAX_FILTER_TAP 12

typedef int16_t InterpKernel12[MAX_FILTER_TAP];

// 12-tap filter (used by the encoder only).
DECLARE_ALIGNED(256, static const InterpKernel12,
                sub_pel_filters_12[SUBPEL_SHIFTS]) = {
  { 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0 },
  { 0, 1, -2, 3, -7, 127, 8, -4, 2, -1, 1, 0 },
  { -1, 2, -3, 6, -13, 124, 18, -8, 4, -2, 2, -1 },
  { -1, 3, -4, 8, -18, 120, 28, -12, 7, -4, 2, -1 },
  { -1, 3, -6, 10, -21, 115, 38, -15, 8, -5, 3, -1 },
  { -2, 4, -6, 12, -24, 108, 49, -18, 10, -6, 3, -2 },
  { -2, 4, -7, 13, -25, 100, 60, -21, 11, -7, 4, -2 },
  { -2, 4, -7, 13, -26, 91, 71, -24, 13, -7, 4, -2 },
  { -2, 4, -7, 13, -25, 81, 81, -25, 13, -7, 4, -2 },
  { -2, 4, -7, 13, -24, 71, 91, -26, 13, -7, 4, -2 },
  { -2, 4, -7, 11, -21, 60, 100, -25, 13, -7, 4, -2 },
  { -2, 3, -6, 10, -18, 49, 108, -24, 12, -6, 4, -2 },
  { -1, 3, -5, 8, -15, 38, 115, -21, 10, -6, 3, -1 },
  { -1, 2, -4, 7, -12, 28, 120, -18, 8, -4, 3, -1 },
  { -1, 2, -2, 4, -8, 18, 124, -13, 6, -3, 2, -1 },
  { 0, 1, -1, 2, -4, 8, 127, -7, 3, -2, 1, 0 }
};

void vp9_temporal_filter_init(void);
void vp9_temporal_filter(struct VP9_COMP *cpi, int distance);

void vp9_temporal_filter_iterate_row_c(struct VP9_COMP *cpi,
                                       struct ThreadData *td, int mb_row,
                                       int mb_col_start, int mb_col_end);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_TEMPORAL_FILTER_H_
