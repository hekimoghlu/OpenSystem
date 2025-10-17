/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#include "vp9/encoder/vp9_treewriter.h"

static void tree2tok(struct vp9_token *tokens, const vpx_tree_index *tree,
                     int i, int v, int l) {
  v += v;
  ++l;

  do {
    const vpx_tree_index j = tree[i++];
    if (j <= 0) {
      tokens[-j].value = v;
      tokens[-j].len = l;
    } else {
      tree2tok(tokens, tree, j, v, l);
    }
  } while (++v & 1);
}

void vp9_tokens_from_tree(struct vp9_token *tokens,
                          const vpx_tree_index *tree) {
  tree2tok(tokens, tree, 0, 0, 0);
}

static unsigned int convert_distribution(unsigned int i, vpx_tree tree,
                                         unsigned int branch_ct[][2],
                                         const unsigned int num_events[]) {
  unsigned int left, right;

  if (tree[i] <= 0)
    left = num_events[-tree[i]];
  else
    left = convert_distribution(tree[i], tree, branch_ct, num_events);

  if (tree[i + 1] <= 0)
    right = num_events[-tree[i + 1]];
  else
    right = convert_distribution(tree[i + 1], tree, branch_ct, num_events);

  branch_ct[i >> 1][0] = left;
  branch_ct[i >> 1][1] = right;
  return left + right;
}

void vp9_tree_probs_from_distribution(vpx_tree tree,
                                      unsigned int branch_ct[/* n-1 */][2],
                                      const unsigned int num_events[/* n */]) {
  convert_distribution(0, tree, branch_ct, num_events);
}
