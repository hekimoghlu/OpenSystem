/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#include <assert.h>

#include "av1/common/av1_loopfilter.h"
#include "av1/common/blockd.h"
#include "av1/common/seg_common.h"
#include "av1/common/quant_common.h"

static const int seg_feature_data_signed[SEG_LVL_MAX] = {
  1, 1, 1, 1, 1, 0, 0, 0
};

static const int seg_feature_data_max[SEG_LVL_MAX] = { MAXQ,
                                                       MAX_LOOP_FILTER,
                                                       MAX_LOOP_FILTER,
                                                       MAX_LOOP_FILTER,
                                                       MAX_LOOP_FILTER,
                                                       7,
                                                       0,
                                                       0 };

// These functions provide access to new segment level features.
// Eventually these function may be "optimized out" but for the moment,
// the coding mechanism is still subject to change so these provide a
// convenient single point of change.

void av1_clearall_segfeatures(struct segmentation *seg) {
  av1_zero(seg->feature_data);
  av1_zero(seg->feature_mask);
}

void av1_calculate_segdata(struct segmentation *seg) {
  seg->segid_preskip = 0;
  seg->last_active_segid = 0;
  for (int i = 0; i < MAX_SEGMENTS; i++) {
    for (int j = 0; j < SEG_LVL_MAX; j++) {
      if (seg->feature_mask[i] & (1 << j)) {
        seg->segid_preskip |= (j >= SEG_LVL_REF_FRAME);
        seg->last_active_segid = i;
      }
    }
  }
}

void av1_enable_segfeature(struct segmentation *seg, int segment_id,
                           SEG_LVL_FEATURES feature_id) {
  seg->feature_mask[segment_id] |= 1 << feature_id;
}

int av1_seg_feature_data_max(SEG_LVL_FEATURES feature_id) {
  return seg_feature_data_max[feature_id];
}

int av1_is_segfeature_signed(SEG_LVL_FEATURES feature_id) {
  return seg_feature_data_signed[feature_id];
}

// The 'seg_data' given for each segment can be either deltas (from the default
// value chosen for the frame) or absolute values.
//
// Valid range for abs values is (0-127 for MB_LVL_ALT_Q), (0-63 for
// SEGMENT_ALT_LF)
// Valid range for delta values are (+/-127 for MB_LVL_ALT_Q), (+/-63 for
// SEGMENT_ALT_LF)
//
// abs_delta = SEGMENT_DELTADATA (deltas) abs_delta = SEGMENT_ABSDATA (use
// the absolute values given).

void av1_set_segdata(struct segmentation *seg, int segment_id,
                     SEG_LVL_FEATURES feature_id, int seg_data) {
  if (seg_data < 0) {
    assert(seg_feature_data_signed[feature_id]);
    assert(-seg_data <= seg_feature_data_max[feature_id]);
  } else {
    assert(seg_data <= seg_feature_data_max[feature_id]);
  }

  seg->feature_data[segment_id][feature_id] = seg_data;
}

// TBD? Functions to read and write segment data with range / validity checking
