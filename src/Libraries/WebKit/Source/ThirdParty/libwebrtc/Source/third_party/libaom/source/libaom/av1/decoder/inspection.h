/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef AOM_AV1_DECODER_INSPECTION_H_
#define AOM_AV1_DECODER_INSPECTION_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "av1/common/seg_common.h"
#if CONFIG_ACCOUNTING
#include "av1/decoder/accounting.h"
#endif

#ifndef AOM_AOM_AOMDX_H_
typedef void (*aom_inspect_cb)(void *decoder, void *data);
#endif

typedef struct insp_mv insp_mv;

struct insp_mv {
  int16_t row;
  int16_t col;
};

typedef struct insp_mi_data insp_mi_data;

struct insp_mi_data {
  insp_mv mv[2];
  int16_t ref_frame[2];
  int16_t mode;
  int16_t uv_mode;
  int16_t bsize;
  int16_t skip;
  int16_t segment_id;
  int16_t dual_filter_type;
  int16_t filter[2];
  int16_t tx_type;
  int16_t tx_size;
  int16_t cdef_level;
  int16_t cdef_strength;
  int16_t cfl_alpha_idx;
  int16_t cfl_alpha_sign;
  int16_t current_qindex;
  int16_t compound_type;
  int16_t motion_mode;
  int16_t intrabc;
  int16_t palette;
  int16_t uv_palette;
};

typedef struct insp_frame_data insp_frame_data;

struct insp_frame_data {
#if CONFIG_ACCOUNTING
  Accounting *accounting;
#endif
  insp_mi_data *mi_grid;
  int16_t frame_number;
  int show_frame;
  int frame_type;
  int base_qindex;
  int mi_rows;
  int mi_cols;
  int tile_mi_rows;
  int tile_mi_cols;
  int16_t y_dequant[MAX_SEGMENTS][2];
  int16_t u_dequant[MAX_SEGMENTS][2];
  int16_t v_dequant[MAX_SEGMENTS][2];
  // TODO(negge): add per frame CDEF data
  int delta_q_present_flag;
  int delta_q_res;
  int show_existing_frame;
};

void ifd_init(insp_frame_data *fd, int frame_width, int frame_height);
void ifd_clear(insp_frame_data *fd);
int ifd_inspect(insp_frame_data *fd, void *decoder, int skip_not_transform);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // AOM_AV1_DECODER_INSPECTION_H_
