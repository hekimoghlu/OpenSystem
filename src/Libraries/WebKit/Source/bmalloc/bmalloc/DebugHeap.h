/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
#pragma once

#include "Environment.h"
#include "FailureAction.h"
#include "Mutex.h"
#include "StaticPerProcess.h"
#include <mutex>
#include <unordered_map>

#if BOS(DARWIN)
#include <malloc/malloc.h>
#endif

namespace bmalloc {
    
class DebugHeap : private StaticPerProcess<DebugHeap> {
public:
    DebugHeap(const LockHolder&);
    
    void* malloc(size_t, FailureAction);
    void* memalign(size_t alignment, size_t, FailureAction);
    void* realloc(void*, size_t, FailureAction);
    void free(void*);
    
    void* memalignLarge(size_t alignment, size_t);
    void freeLarge(void* base);

#if BENABLE(MALLOC_SIZE)
    size_t mallocSize(const void* object) { return malloc_size(object); }
#endif

#if BENABLE(MALLOC_GOOD_SIZE)
    size_t mallocGoodSize(size_t size) { return malloc_good_size(size); }
#endif

    void scavenge();
    void dump();

    static DebugHeap* tryGet();
    static DebugHeap* getExisting();

private:
    static DebugHeap* tryGetSlow();
    
#if BOS(DARWIN)
    malloc_zone_t* m_zone;
#endif
    
    // This is the debug heap. We can use whatever data structures we like. It doesn't matter.
    size_t m_pageSize { 0 };
    std::unordered_map<void*, size_t> m_sizeMap;
};
BALLOW_DEPRECATED_DECLARATIONS_BEGIN
DECLARE_STATIC_PER_PROCESS_STORAGE(DebugHeap);
BALLOW_DEPRECATED_DECLARATIONS_END

extern BEXPORT DebugHeap* debugHeapCache;

BINLINE DebugHeap* debugHeapDisabled()
{
    return reinterpret_cast<DebugHeap*>(static_cast<uintptr_t>(1));
}

BINLINE DebugHeap* DebugHeap::tryGet()
{
    DebugHeap* result = debugHeapCache;
    if (result == debugHeapDisabled())
        return nullptr;
    if (result)
        return result;
    return tryGetSlow();
}

BINLINE DebugHeap* DebugHeap::getExisting()
{
    DebugHeap* result = tryGet();
    RELEASE_BASSERT(result);
    return result;
}

} // namespace bmalloc
