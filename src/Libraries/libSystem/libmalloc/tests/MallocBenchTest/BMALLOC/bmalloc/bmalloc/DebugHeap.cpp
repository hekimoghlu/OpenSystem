/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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
#include "DebugHeap.h"

#include "Algorithm.h"
#include "BAssert.h"
#include "BPlatform.h"
#include "VMAllocate.h"
#include <cstdlib>
#include <thread>

namespace bmalloc {
    
#if BOS(DARWIN)

DebugHeap::DebugHeap(std::lock_guard<Mutex>&)
    : m_zone(malloc_create_zone(0, 0))
    , m_pageSize(vmPageSize())
{
    malloc_set_zone_name(m_zone, "WebKit Using System Malloc");
}

void* DebugHeap::malloc(size_t size)
{
    void* result = malloc_zone_malloc(m_zone, size);
    if (!result)
        BCRASH();
    return result;
}

void* DebugHeap::memalign(size_t alignment, size_t size, bool crashOnFailure)
{
    void* result = malloc_zone_memalign(m_zone, alignment, size);
    if (!result && crashOnFailure)
        BCRASH();
    return result;
}

void* DebugHeap::realloc(void* object, size_t size, bool crashOnFailure)
{
    void* result = malloc_zone_realloc(m_zone, object, size);
    if (!result && crashOnFailure)
        BCRASH();
    return result;
}

void DebugHeap::free(void* object)
{
    malloc_zone_free(m_zone, object);
}

#else

DebugHeap::DebugHeap(std::lock_guard<Mutex>&)
    : m_pageSize(vmPageSize())
{
}

void* DebugHeap::malloc(size_t size)
{
    void* result = ::malloc(size);
    if (!result)
        BCRASH();
    return result;
}

void* DebugHeap::memalign(size_t alignment, size_t size, bool crashOnFailure)
{
    void* result;
    if (posix_memalign(&result, alignment, size)) {
        if (crashOnFailure)
            BCRASH();
        return nullptr;
    }
    return result;
}

void* DebugHeap::realloc(void* object, size_t size, bool crashOnFailure)
{
    void* result = ::realloc(object, size);
    if (!result && crashOnFailure)
        BCRASH();
    return result;
}

void DebugHeap::free(void* object)
{
    ::free(object);
}
    
#endif

// FIXME: This looks an awful lot like the code in wtf/Gigacage.cpp for large allocation.
// https://bugs.webkit.org/show_bug.cgi?id=175086

void* DebugHeap::memalignLarge(size_t alignment, size_t size)
{
    alignment = roundUpToMultipleOf(m_pageSize, alignment);
    size = roundUpToMultipleOf(m_pageSize, size);
    void* result = tryVMAllocate(alignment, size);
    if (!result)
        return nullptr;
    {
        std::lock_guard<std::mutex> locker(m_lock);
        m_sizeMap[result] = size;
    }
    return result;
}

void DebugHeap::freeLarge(void* base)
{
    if (!base)
        return;
    
    size_t size;
    {
        std::lock_guard<std::mutex> locker(m_lock);
        size = m_sizeMap[base];
        size_t numErased = m_sizeMap.erase(base);
        RELEASE_BASSERT(numErased == 1);
    }
    
    vmDeallocate(base, size);
}

} // namespace bmalloc
