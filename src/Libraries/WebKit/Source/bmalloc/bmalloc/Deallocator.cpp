/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#include "BAssert.h"
#include "BInline.h"
#include "Chunk.h"
#include "Deallocator.h"
#include "Environment.h"
#include "Heap.h"
#include "Object.h"
#include "PerProcess.h"
#include <algorithm>
#include <cstdlib>
#include <sys/mman.h>

#if !BUSE(LIBPAS)

namespace bmalloc {

Deallocator::Deallocator(Heap& heap)
    : m_heap(heap)
{
    BASSERT(!Environment::get()->isDebugHeapEnabled());
}

Deallocator::~Deallocator()
{
    scavenge();
}
    
void Deallocator::scavenge()
{
    UniqueLockHolder lock(Heap::mutex());

    processObjectLog(lock);
    m_heap.deallocateLineCache(lock, lineCache(lock));
}

void Deallocator::processObjectLog(UniqueLockHolder& lock)
{
    for (Object object : m_objectLog)
        m_heap.derefSmallLine(lock, object, lineCache(lock));
    m_objectLog.clear();
}

void Deallocator::deallocateSlowCase(void* object)
{
    if (!object)
        return;

    if (m_heap.isLarge(object)) {
        UniqueLockHolder lock(Heap::mutex());
        m_heap.deallocateLarge(lock, object);
        return;
    }

    if (m_objectLog.size() == m_objectLog.capacity()) {
        UniqueLockHolder lock(Heap::mutex());
        processObjectLog(lock);
    }

    m_objectLog.push(object);
}

} // namespace bmalloc

#endif
