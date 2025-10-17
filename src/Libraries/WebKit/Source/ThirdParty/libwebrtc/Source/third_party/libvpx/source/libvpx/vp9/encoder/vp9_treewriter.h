/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#ifndef VPX_VP9_ENCODER_VP9_TREEWRITER_H_
#define VPX_VP9_ENCODER_VP9_TREEWRITER_H_

#include "vpx_dsp/bitwriter.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_tree_probs_from_distribution(vpx_tree tree,
                                      unsigned int branch_ct[/* n - 1 */][2],
                                      const unsigned int num_events[/* n */]);

struct vp9_token {
  int value;
  int len;
};

void vp9_tokens_from_tree(struct vp9_token *, const vpx_tree_index *);

static INLINE void vp9_write_tree(vpx_writer *w, const vpx_tree_index *tree,
                                  const vpx_prob *probs, int bits, int len,
                                  vpx_tree_index i) {
  do {
    const int bit = (bits >> --len) & 1;
    vpx_write(w, bit, probs[i >> 1]);
    i = tree[i + bit];
  } while (len);
}

static INLINE void vp9_write_token(vpx_writer *w, const vpx_tree_index *tree,
                                   const vpx_prob *probs,
                                   const struct vp9_token *token) {
  vp9_write_tree(w, tree, probs, token->value, token->len, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_TREEWRITER_H_
