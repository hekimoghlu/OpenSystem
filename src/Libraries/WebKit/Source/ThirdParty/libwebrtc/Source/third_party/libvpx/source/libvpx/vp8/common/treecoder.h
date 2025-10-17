/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#ifndef VPX_VP8_COMMON_TREECODER_H_
#define VPX_VP8_COMMON_TREECODER_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char vp8bc_index_t; /* probability index */

typedef unsigned char vp8_prob;

#define vp8_prob_half ((vp8_prob)128)

typedef signed char vp8_tree_index;
struct bool_coder_spec;

typedef struct bool_coder_spec bool_coder_spec;
typedef struct bool_writer bool_writer;
typedef struct bool_reader bool_reader;

typedef const bool_coder_spec c_bool_coder_spec;
typedef const bool_writer c_bool_writer;
typedef const bool_reader c_bool_reader;

#define vp8_complement(x) (255 - (x))

/* We build coding trees compactly in arrays.
   Each node of the tree is a pair of vp8_tree_indices.
   Array index often references a corresponding probability table.
   Index <= 0 means done encoding/decoding and value = -Index,
   Index > 0 means need another bit, specification at index.
   Nonnegative indices are always even;  processing begins at node 0. */

typedef const vp8_tree_index vp8_tree[], *vp8_tree_p;

typedef const struct vp8_token_struct {
  int value;
  int Len;
} vp8_token;

/* Construct encoding array from tree. */

void vp8_tokens_from_tree(struct vp8_token_struct *, vp8_tree);
void vp8_tokens_from_tree_offset(struct vp8_token_struct *, vp8_tree,
                                 int offset);

/* Convert array of token occurrence counts into a table of probabilities
   for the associated binary encoding tree.  Also writes count of branches
   taken for each node on the tree; this facilitiates decisions as to
   probability updates. */

void vp8_tree_probs_from_distribution(int n, /* n = size of alphabet */
                                      vp8_token tok[/* n */], vp8_tree tree,
                                      vp8_prob probs[/* n-1 */],
                                      unsigned int branch_ct[/* n-1 */][2],
                                      const unsigned int num_events[/* n */],
                                      unsigned int Pfactor, int Round);

/* Variant of above using coder spec rather than hardwired 8-bit probs. */

void vp8bc_tree_probs_from_distribution(int n, /* n = size of alphabet */
                                        vp8_token tok[/* n */], vp8_tree tree,
                                        vp8_prob probs[/* n-1 */],
                                        unsigned int branch_ct[/* n-1 */][2],
                                        const unsigned int num_events[/* n */],
                                        c_bool_coder_spec *s);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_TREECODER_H_
