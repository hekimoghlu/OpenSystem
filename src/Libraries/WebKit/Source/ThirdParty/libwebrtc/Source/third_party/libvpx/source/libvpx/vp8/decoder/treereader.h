/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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
#ifndef VPX_VP8_DECODER_TREEREADER_H_
#define VPX_VP8_DECODER_TREEREADER_H_

#include "./vpx_config.h"
#include "vp8/common/treecoder.h"
#include "dboolhuff.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef BOOL_DECODER vp8_reader;

#define vp8_read vp8dx_decode_bool
#define vp8_read_literal vp8_decode_value
#define vp8_read_bit(R) vp8_read(R, vp8_prob_half)

/* Intent of tree data structure is to make decoding trivial. */

static INLINE int vp8_treed_read(
    vp8_reader *const r, /* !!! must return a 0 or 1 !!! */
    vp8_tree t, const vp8_prob *const p) {
  vp8_tree_index i = 0;

  while ((i = t[i + vp8_read(r, p[i >> 1])]) > 0) {
  }

  return -i;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_DECODER_TREEREADER_H_
