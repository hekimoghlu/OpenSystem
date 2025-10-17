/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#include <wtf/DebugHeap.h>

#if ENABLE(MALLOC_HEAP_BREAKDOWN)

#include <cstdlib>
#include <thread>

namespace WTF {

DebugHeap::DebugHeap(const char* heapName)
    : m_zone(malloc_create_zone(0, 0))
{
    malloc_set_zone_name(m_zone, heapName);
}

void* DebugHeap::malloc(size_t size)
{
    void* result = malloc_zone_malloc(m_zone, size);
    if (!result)
        CRASH();
    return result;
}

void* DebugHeap::calloc(size_t numElements, size_t elementSize)
{
    void* result = malloc_zone_calloc(m_zone, numElements, elementSize);
    if (!result)
        CRASH();
    return result;
}

void* DebugHeap::memalign(size_t alignment, size_t size, bool crashOnFailure)
{
    void* result = malloc_zone_memalign(m_zone, alignment, size);
    if (!result && crashOnFailure)
        CRASH();
    return result;
}

void* DebugHeap::realloc(void* object, size_t size)
{
    void* result = malloc_zone_realloc(m_zone, object, size);
    if (!result)
        CRASH();
    return result;
}

void DebugHeap::free(void* object)
{
    malloc_zone_free(m_zone, object);
}

} // namespace WTF

#endif // ENABLE(MALLOC_HEAP_BREAKDOWN)
