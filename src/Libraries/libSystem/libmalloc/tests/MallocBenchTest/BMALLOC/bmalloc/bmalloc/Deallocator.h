/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
#ifndef Deallocator_h
#define Deallocator_h

#include "BExport.h"
#include "FixedVector.h"
#include "SmallPage.h"
#include <mutex>

namespace bmalloc {

class DebugHeap;
class Heap;
class Mutex;

// Per-cache object deallocator.

class Deallocator {
public:
    Deallocator(Heap&);
    ~Deallocator();

    void deallocate(void*);
    void scavenge();
    
    void processObjectLog(std::unique_lock<Mutex>&);
    
    LineCache& lineCache(std::unique_lock<Mutex>&) { return m_lineCache; }

private:
    bool deallocateFastCase(void*);
    BEXPORT void deallocateSlowCase(void*);

    Heap& m_heap;
    FixedVector<void*, deallocatorLogCapacity> m_objectLog;
    LineCache m_lineCache; // The Heap removes items from this cache.
    DebugHeap* m_debugHeap;
};

inline bool Deallocator::deallocateFastCase(void* object)
{
    BASSERT(mightBeLarge(nullptr));
    if (mightBeLarge(object))
        return false;

    if (m_objectLog.size() == m_objectLog.capacity())
        return false;

    m_objectLog.push(object);
    return true;
}

inline void Deallocator::deallocate(void* object)
{
    if (!deallocateFastCase(object))
        deallocateSlowCase(object);
}

} // namespace bmalloc

#endif // Deallocator_h
