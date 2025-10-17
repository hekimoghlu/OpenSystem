/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
/*
 * sync.c   the Netwide Disassembler synchronisation processing module
 */

#include "compiler.h"


#include "nasmlib.h"
#include "sync.h"

#define SYNC_MAX_SHIFT          31
#define SYNC_MAX_SIZE           (1U << SYNC_MAX_SHIFT)

/* initial # of sync points (*must* be power of two)*/
#define SYNC_INITIAL_CHUNK      (1U << 12)

/*
 * This lot manages the current set of sync points by means of a
 * heap (priority queue) structure.
 */

static struct Sync {
    uint64_t pos;
    uint32_t length;
} *synx;

static uint32_t max_synx, nsynx;

static inline void swap_sync(uint32_t dst, uint32_t src)
{
    struct Sync t = synx[dst];
    synx[dst] = synx[src];
    synx[src] = t;
}

void init_sync(void)
{
    max_synx = SYNC_INITIAL_CHUNK;
    synx = nasm_malloc((max_synx + 1) * sizeof(*synx));
    nsynx = 0;
}

void add_sync(uint64_t pos, uint32_t length)
{
    uint32_t i;

    if (nsynx >= max_synx) {
        if (max_synx >= SYNC_MAX_SIZE) /* too many sync points! */
            return;
        max_synx = (max_synx << 1);
        synx = nasm_realloc(synx, (max_synx + 1) * sizeof(*synx));
    }

    nsynx++;
    synx[nsynx].pos = pos;
    synx[nsynx].length = length;

    for (i = nsynx; i > 1; i /= 2) {
        if (synx[i / 2].pos > synx[i].pos)
            swap_sync(i / 2, i);
    }
}

uint64_t next_sync(uint64_t position, uint32_t *length)
{
    while (nsynx > 0 && synx[1].pos + synx[1].length <= position) {
        uint32_t i, j;

        swap_sync(nsynx, 1);
        nsynx--;

        i = 1;
        while (i * 2 <= nsynx) {
            j = i * 2;
            if (synx[j].pos < synx[i].pos &&
                (j + 1 > nsynx || synx[j + 1].pos > synx[j].pos)) {
                swap_sync(j, i);
                i = j;
            } else if (j + 1 <= nsynx && synx[j + 1].pos < synx[i].pos) {
                swap_sync(j + 1, i);
                i = j + 1;
            } else
                break;
        }
    }

    if (nsynx > 0) {
        if (length)
            *length = synx[1].length;
        return synx[1].pos;
    } else {
        if (length)
            *length = 0L;
        return 0;
    }
}
