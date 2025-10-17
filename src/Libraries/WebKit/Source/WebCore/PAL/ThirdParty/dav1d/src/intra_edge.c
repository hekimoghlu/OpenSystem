/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#include "config.h"

#include <stdlib.h>

#include "common/attributes.h"

#include "src/intra_edge.h"
#include "src/levels.h"

struct ModeSelMem {
    EdgeBranch *nwc[3 /* 64x64, 32x32, 16x16 */];
    EdgeTip *nt;
};

static void init_edges(EdgeNode *const node,
                       const enum BlockLevel bl,
                       const enum EdgeFlags edge_flags)
{
    node->o = edge_flags;

#define ALL_FL(t) (EDGE_I444_##t | EDGE_I422_##t | EDGE_I420_##t)
    if (bl == BL_8X8) {
        EdgeTip *const nt = (EdgeTip *) node;

        node->h[0] = edge_flags | ALL_FL(LEFT_HAS_BOTTOM);
        node->h[1] = edge_flags & (ALL_FL(LEFT_HAS_BOTTOM) |
                                   EDGE_I420_TOP_HAS_RIGHT);

        node->v[0] = edge_flags | ALL_FL(TOP_HAS_RIGHT);
        node->v[1] = edge_flags & (ALL_FL(TOP_HAS_RIGHT) |
                                   EDGE_I420_LEFT_HAS_BOTTOM |
                                   EDGE_I422_LEFT_HAS_BOTTOM);

        nt->split[0] = ALL_FL(TOP_HAS_RIGHT) | ALL_FL(LEFT_HAS_BOTTOM);
        nt->split[1] = (edge_flags & ALL_FL(TOP_HAS_RIGHT)) |
                       EDGE_I422_LEFT_HAS_BOTTOM;
        nt->split[2] = edge_flags | EDGE_I444_TOP_HAS_RIGHT;
        nt->split[3] = edge_flags & (EDGE_I420_TOP_HAS_RIGHT |
                                     EDGE_I420_LEFT_HAS_BOTTOM |
                                     EDGE_I422_LEFT_HAS_BOTTOM);
    } else {
        EdgeBranch *const nwc = (EdgeBranch *) node;

        node->h[0] = edge_flags | ALL_FL(LEFT_HAS_BOTTOM);
        node->h[1] = edge_flags & ALL_FL(LEFT_HAS_BOTTOM);

        node->v[0] = edge_flags | ALL_FL(TOP_HAS_RIGHT);
        node->v[1] = edge_flags & ALL_FL(TOP_HAS_RIGHT);

        nwc->h4[0] = edge_flags | ALL_FL(LEFT_HAS_BOTTOM);
        nwc->h4[1] =
        nwc->h4[2] = ALL_FL(LEFT_HAS_BOTTOM);
        nwc->h4[3] = edge_flags & ALL_FL(LEFT_HAS_BOTTOM);
        if (bl == BL_16X16)
            nwc->h4[1] |= edge_flags & EDGE_I420_TOP_HAS_RIGHT;

        nwc->v4[0] = edge_flags | ALL_FL(TOP_HAS_RIGHT);
        nwc->v4[1] =
        nwc->v4[2] = ALL_FL(TOP_HAS_RIGHT);
        nwc->v4[3] = edge_flags & ALL_FL(TOP_HAS_RIGHT);
        if (bl == BL_16X16)
            nwc->v4[1] |= edge_flags & (EDGE_I420_LEFT_HAS_BOTTOM |
                                        EDGE_I422_LEFT_HAS_BOTTOM);

        nwc->tls[0] = ALL_FL(TOP_HAS_RIGHT) | ALL_FL(LEFT_HAS_BOTTOM);
        nwc->tls[1] = edge_flags & ALL_FL(LEFT_HAS_BOTTOM);
        nwc->tls[2] = edge_flags & ALL_FL(TOP_HAS_RIGHT);

        nwc->trs[0] = edge_flags | ALL_FL(TOP_HAS_RIGHT);
        nwc->trs[1] = edge_flags | ALL_FL(LEFT_HAS_BOTTOM);
        nwc->trs[2] = 0;

        nwc->tts[0] = ALL_FL(TOP_HAS_RIGHT) | ALL_FL(LEFT_HAS_BOTTOM);
        nwc->tts[1] = edge_flags & ALL_FL(TOP_HAS_RIGHT);
        nwc->tts[2] = edge_flags & ALL_FL(LEFT_HAS_BOTTOM);

        nwc->tbs[0] = edge_flags | ALL_FL(LEFT_HAS_BOTTOM);
        nwc->tbs[1] = edge_flags | ALL_FL(TOP_HAS_RIGHT);
        nwc->tbs[2] = 0;
    }
}

static void init_mode_node(EdgeBranch *const nwc,
                           const enum BlockLevel bl,
                           struct ModeSelMem *const mem,
                           const int top_has_right,
                           const int left_has_bottom)
{
    init_edges(&nwc->node, bl,
               (top_has_right ? ALL_FL(TOP_HAS_RIGHT) : 0) |
               (left_has_bottom ? ALL_FL(LEFT_HAS_BOTTOM) : 0));
    if (bl == BL_16X16) {
        for (int n = 0; n < 4; n++) {
            EdgeTip *const nt = mem->nt++;
            nwc->split[n] = &nt->node;
            init_edges(&nt->node, bl + 1,
                       ((n == 3 || (n == 1 && !top_has_right)) ? 0 :
                        ALL_FL(TOP_HAS_RIGHT)) |
                       (!(n == 0 || (n == 2 && left_has_bottom)) ? 0 :
                        ALL_FL(LEFT_HAS_BOTTOM)));
        }
    } else {
        for (int n = 0; n < 4; n++) {
            EdgeBranch *const nwc_child = mem->nwc[bl]++;
            nwc->split[n] = &nwc_child->node;
            init_mode_node(nwc_child, bl + 1, mem,
                           !(n == 3 || (n == 1 && !top_has_right)),
                           n == 0 || (n == 2 && left_has_bottom));
        }
    }
}

void dav1d_init_mode_tree(EdgeNode *const root_node, EdgeTip *const nt,
                          const int allow_sb128)
{
    EdgeBranch *const root = (EdgeBranch *) root_node;
    struct ModeSelMem mem;
    mem.nt = nt;

    if (allow_sb128) {
        mem.nwc[BL_128X128] = &root[1];
        mem.nwc[BL_64X64] = &root[1 + 4];
        mem.nwc[BL_32X32] = &root[1 + 4 + 16];
        init_mode_node(root, BL_128X128, &mem, 1, 0);
        assert(mem.nwc[BL_128X128] == &root[1 + 4]);
        assert(mem.nwc[BL_64X64] == &root[1 + 4 + 16]);
        assert(mem.nwc[BL_32X32] == &root[1 + 4 + 16 + 64]);
        assert(mem.nt == &nt[256]);
    } else {
        mem.nwc[BL_128X128] = NULL;
        mem.nwc[BL_64X64] = &root[1];
        mem.nwc[BL_32X32] = &root[1 + 4];
        init_mode_node(root, BL_64X64, &mem, 1, 0);
        assert(mem.nwc[BL_64X64] == &root[1 + 4]);
        assert(mem.nwc[BL_32X32] == &root[1 + 4 + 16]);
        assert(mem.nt == &nt[64]);
    }
}
