/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#include "treewriter.h"

static void cost(int *const C, vp8_tree T, const vp8_prob *const P, int i,
                 int c) {
  const vp8_prob p = P[i >> 1];

  do {
    const vp8_tree_index j = T[i];
    const int d = c + vp8_cost_bit(p, i & 1);

    if (j <= 0) {
      C[-j] = d;
    } else {
      cost(C, T, P, j, d);
    }
  } while (++i & 1);
}
void vp8_cost_tokens(int *c, const vp8_prob *p, vp8_tree t) {
  cost(c, t, p, 0, 0);
}
void vp8_cost_tokens2(int *c, const vp8_prob *p, vp8_tree t, int start) {
  cost(c, t, p, start, 0);
}
