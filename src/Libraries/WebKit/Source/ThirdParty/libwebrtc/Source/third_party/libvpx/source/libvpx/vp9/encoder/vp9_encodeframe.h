/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#ifndef VPX_VP9_ENCODER_VP9_ENCODEFRAME_H_
#define VPX_VP9_ENCODER_VP9_ENCODEFRAME_H_

#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct macroblock;
struct yv12_buffer_config;
struct VP9_COMP;
struct ThreadData;

// Constants used in SOURCE_VAR_BASED_PARTITION
#define VAR_HIST_MAX_BG_VAR 1000
#define VAR_HIST_FACTOR 10
#define VAR_HIST_BINS (VAR_HIST_MAX_BG_VAR / VAR_HIST_FACTOR + 1)
#define VAR_HIST_LARGE_CUT_OFF 75
#define VAR_HIST_SMALL_CUT_OFF 45

void vp9_setup_src_planes(struct macroblock *x,
                          const struct yv12_buffer_config *src, int mi_row,
                          int mi_col);

void vp9_encode_frame(struct VP9_COMP *cpi);

void vp9_init_tile_data(struct VP9_COMP *cpi);
void vp9_encode_tile(struct VP9_COMP *cpi, struct ThreadData *td, int tile_row,
                     int tile_col);

void vp9_encode_sb_row(struct VP9_COMP *cpi, struct ThreadData *td,
                       int tile_row, int tile_col, int mi_row);

void vp9_set_variance_partition_thresholds(struct VP9_COMP *cpi, int q,
                                           int content_state);

struct KMEANS_DATA;
void vp9_kmeans(double *ctr_ls, double *boundary_ls, int *count_ls, int k,
                struct KMEANS_DATA *arr, int size);
int vp9_get_group_idx(double value, double *boundary_ls, int k);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_ENCODEFRAME_H_
