/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
#include <wtf/HashTable.h>

#include <wtf/NeverDestroyed.h>

namespace WTF {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(HashTable);

#if DUMP_HASHTABLE_STATS

std::atomic<unsigned> HashTableStats::numAccesses;
std::atomic<unsigned> HashTableStats::numRehashes;
std::atomic<unsigned> HashTableStats::numRemoves;
std::atomic<unsigned> HashTableStats::numReinserts;

unsigned HashTableStats::numCollisions;
unsigned HashTableStats::collisionGraph[4096];
unsigned HashTableStats::maxCollisions;

static Lock hashTableStatsMutex;

void HashTableStats::recordCollisionAtCount(unsigned count)
{
    Locker locker { hashTableStatsMutex };

    if (count > maxCollisions)
        maxCollisions = count;
    numCollisions++;
    collisionGraph[count]++;
}

void HashTableStats::dumpStats()
{
    Locker locker { hashTableStatsMutex };

    dataLogF("\nWTF::HashTable statistics\n\n");
    dataLogF("%u accesses\n", numAccesses.load());
    dataLogF("%d total collisions, average %.2f probes per access\n", numCollisions, 1.0 * (numAccesses + numCollisions) / numAccesses);
    dataLogF("longest collision chain: %d\n", maxCollisions);
    for (unsigned i = 1; i <= maxCollisions; i++) {
        dataLogF("  %u lookups with exactly %u collisions (%.2f%% , %.2f%% with this many or more)\n", collisionGraph[i], i, 100.0 * (collisionGraph[i] - collisionGraph[i+1]) / numAccesses, 100.0 * collisionGraph[i] / numAccesses);
    }
    dataLogF("%d rehashes\n", numRehashes.load());
    dataLogF("%d reinserts\n", numReinserts.load());
}

#endif

} // namespace WTF
