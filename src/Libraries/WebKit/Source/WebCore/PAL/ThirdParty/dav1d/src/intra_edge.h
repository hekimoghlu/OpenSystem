/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#ifndef DAV1D_SRC_INTRA_EDGE_H
#define DAV1D_SRC_INTRA_EDGE_H

enum EdgeFlags {
    EDGE_I444_TOP_HAS_RIGHT = 1 << 0,
    EDGE_I422_TOP_HAS_RIGHT = 1 << 1,
    EDGE_I420_TOP_HAS_RIGHT = 1 << 2,
    EDGE_I444_LEFT_HAS_BOTTOM = 1 << 3,
    EDGE_I422_LEFT_HAS_BOTTOM = 1 << 4,
    EDGE_I420_LEFT_HAS_BOTTOM = 1 << 5,
};

typedef struct EdgeNode EdgeNode;
struct EdgeNode {
    enum EdgeFlags o, h[2], v[2];
};
typedef struct EdgeTip {
    EdgeNode node;
    enum EdgeFlags split[4];
} EdgeTip;
typedef struct EdgeBranch {
    EdgeNode node;
    enum EdgeFlags tts[3], tbs[3], tls[3], trs[3], h4[4], v4[4];
    EdgeNode *split[4];
} EdgeBranch;

void dav1d_init_mode_tree(EdgeNode *const root, EdgeTip *const nt,
                          const int allow_sb128);

#endif /* DAV1D_SRC_INTRA_EDGE_H */
