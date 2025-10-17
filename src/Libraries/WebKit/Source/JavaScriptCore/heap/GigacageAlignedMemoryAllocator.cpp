/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
#include "GigacageAlignedMemoryAllocator.h"

#if ENABLE(MALLOC_HEAP_BREAKDOWN)
#include <wtf/text/MakeString.h>
#endif

#include <wtf/text/StringView.h>

namespace JSC {

GigacageAlignedMemoryAllocator::GigacageAlignedMemoryAllocator(Gigacage::Kind kind)
    : m_kind(kind)
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    , m_heap(makeString("GigacageAlignedMemoryAllocator "_s, m_kind).utf8().data())
#endif
{
}

GigacageAlignedMemoryAllocator::~GigacageAlignedMemoryAllocator() = default;

void* GigacageAlignedMemoryAllocator::tryAllocateAlignedMemory(size_t alignment, size_t size)
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    return m_heap.memalign(alignment, size, true);
#else
    return Gigacage::tryAlignedMalloc(m_kind, alignment, size);
#endif
}

void GigacageAlignedMemoryAllocator::freeAlignedMemory(void* basePtr)
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    return m_heap.free(basePtr);
#else
    Gigacage::alignedFree(m_kind, basePtr);
#endif
}

void GigacageAlignedMemoryAllocator::dump(PrintStream& out) const
{
    out.print(m_kind, "Gigacage");
}

void* GigacageAlignedMemoryAllocator::tryAllocateMemory(size_t size)
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    return m_heap.malloc(size);
#else
    return Gigacage::tryMalloc(m_kind, size);
#endif
}

void GigacageAlignedMemoryAllocator::freeMemory(void* pointer)
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    m_heap.free(pointer);
#else
    Gigacage::free(m_kind, pointer);
#endif
}

void* GigacageAlignedMemoryAllocator::tryReallocateMemory(void* pointer, size_t size)
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    return m_heap.realloc(pointer, size);
#else
    return Gigacage::tryRealloc(m_kind, pointer, size);
#endif
}

} // namespace JSC

