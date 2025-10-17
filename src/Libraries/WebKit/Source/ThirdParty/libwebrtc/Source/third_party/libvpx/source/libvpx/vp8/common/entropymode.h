/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#ifndef VPX_VP8_COMMON_ENTROPYMODE_H_
#define VPX_VP8_COMMON_ENTROPYMODE_H_

#include "onyxc_int.h"
#include "treecoder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SUBMVREF_NORMAL,
  SUBMVREF_LEFT_ZED,
  SUBMVREF_ABOVE_ZED,
  SUBMVREF_LEFT_ABOVE_SAME,
  SUBMVREF_LEFT_ABOVE_ZED
} sumvfref_t;

typedef int vp8_mbsplit[16];

#define VP8_NUMMBSPLITS 4

extern const vp8_mbsplit vp8_mbsplits[VP8_NUMMBSPLITS];

extern const int vp8_mbsplit_count[VP8_NUMMBSPLITS]; /* # of subsets */

extern const vp8_prob vp8_mbsplit_probs[VP8_NUMMBSPLITS - 1];

extern int vp8_mv_cont(const int_mv *l, const int_mv *a);
#define SUBMVREF_COUNT 5
extern const vp8_prob vp8_sub_mv_ref_prob2[SUBMVREF_COUNT][VP8_SUBMVREFS - 1];

extern const unsigned int vp8_kf_default_bmode_counts[VP8_BINTRAMODES]
                                                     [VP8_BINTRAMODES]
                                                     [VP8_BINTRAMODES];

extern const vp8_tree_index vp8_bmode_tree[];

extern const vp8_tree_index vp8_ymode_tree[];
extern const vp8_tree_index vp8_kf_ymode_tree[];
extern const vp8_tree_index vp8_uv_mode_tree[];

extern const vp8_tree_index vp8_mbsplit_tree[];
extern const vp8_tree_index vp8_mv_ref_tree[];
extern const vp8_tree_index vp8_sub_mv_ref_tree[];

extern const struct vp8_token_struct vp8_bmode_encodings[VP8_BINTRAMODES];
extern const struct vp8_token_struct vp8_ymode_encodings[VP8_YMODES];
extern const struct vp8_token_struct vp8_kf_ymode_encodings[VP8_YMODES];
extern const struct vp8_token_struct vp8_uv_mode_encodings[VP8_UV_MODES];
extern const struct vp8_token_struct vp8_mbsplit_encodings[VP8_NUMMBSPLITS];

/* Inter mode values do not start at zero */

extern const struct vp8_token_struct vp8_mv_ref_encoding_array[VP8_MVREFS];
extern const struct vp8_token_struct
    vp8_sub_mv_ref_encoding_array[VP8_SUBMVREFS];

extern const vp8_tree_index vp8_small_mvtree[];

extern const struct vp8_token_struct vp8_small_mvencodings[8];

/* Key frame default mode probs */
extern const vp8_prob vp8_kf_bmode_prob[VP8_BINTRAMODES][VP8_BINTRAMODES]
                                       [VP8_BINTRAMODES - 1];
extern const vp8_prob vp8_kf_uv_mode_prob[VP8_UV_MODES - 1];
extern const vp8_prob vp8_kf_ymode_prob[VP8_YMODES - 1];

void vp8_init_mbmode_probs(VP8_COMMON *x);
void vp8_default_bmode_probs(vp8_prob dest[VP8_BINTRAMODES - 1]);
void vp8_kf_default_bmode_probs(
    vp8_prob dest[VP8_BINTRAMODES][VP8_BINTRAMODES][VP8_BINTRAMODES - 1]);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_ENTROPYMODE_H_
