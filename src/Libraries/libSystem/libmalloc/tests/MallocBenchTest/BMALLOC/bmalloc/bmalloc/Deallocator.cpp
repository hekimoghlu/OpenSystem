/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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
#include "DebugHeap.h"
#include "Heap.h"
#include "Object.h"
#include "PerProcess.h"
#include <algorithm>
#include <cstdlib>
#include <sys/mman.h>

namespace bmalloc {

Deallocator::Deallocator(Heap& heap)
    : m_heap(heap)
    , m_debugHeap(heap.debugHeap())
{
    if (m_debugHeap) {
        // Fill the object log in order to disable the fast path.
        while (m_objectLog.size() != m_objectLog.capacity())
            m_objectLog.push(nullptr);
    }
}

Deallocator::~Deallocator()
{
    scavenge();
}
    
void Deallocator::scavenge()
{
    if (m_debugHeap)
        return;

    std::unique_lock<Mutex> lock(Heap::mutex());

    processObjectLog(lock);
    m_heap.deallocateLineCache(lock, lineCache(lock));
}

void Deallocator::processObjectLog(std::unique_lock<Mutex>& lock)
{
    for (Object object : m_objectLog)
        m_heap.derefSmallLine(lock, object, lineCache(lock));
    m_objectLog.clear();
}

void Deallocator::deallocateSlowCase(void* object)
{
    if (m_debugHeap)
        return m_debugHeap->free(object);

    if (!object)
        return;

    std::unique_lock<Mutex> lock(Heap::mutex());
    if (m_heap.isLarge(lock, object)) {
        m_heap.deallocateLarge(lock, object);
        return;
    }

    if (m_objectLog.size() == m_objectLog.capacity())
        processObjectLog(lock);

    m_objectLog.push(object);
}

} // namespace bmalloc
