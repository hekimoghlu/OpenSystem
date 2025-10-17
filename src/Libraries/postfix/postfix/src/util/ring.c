/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
/* System libraries. */

/* Application-specific. */

#include "ring.h"

/* ring_init - initialize ring head */

void    ring_init(ring)
RING   *ring;
{
    ring->pred = ring->succ = ring;
}

/* ring_append - insert entry after ring head */

void    ring_append(ring, entry)
RING   *ring;
RING   *entry;
{
    entry->succ = ring->succ;
    entry->pred = ring;
    ring->succ->pred = entry;
    ring->succ = entry;
}

/* ring_prepend - insert new entry before ring head */

void    ring_prepend(ring, entry)
RING   *ring;
RING   *entry;
{
    entry->pred = ring->pred;
    entry->succ = ring;
    ring->pred->succ = entry;
    ring->pred = entry;
}

/* ring_detach - remove entry from ring */

void    ring_detach(entry)
RING   *entry;
{
    RING   *succ = entry->succ;
    RING   *pred = entry->pred;

    pred->succ = succ;
    succ->pred = pred;

    entry->succ = entry->pred = 0;
}
