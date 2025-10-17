/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#include "IsoAlignedMemoryAllocator.h"

#include "MarkedBlock.h"

#if ENABLE(MALLOC_HEAP_BREAKDOWN)
#include <wtf/text/MakeString.h>
#endif

namespace JSC {

IsoAlignedMemoryAllocator::IsoAlignedMemoryAllocator(CString name)
    : Base(name)
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    , m_heap(makeString("IsoAlignedAllocator "_s, name.span()).utf8().data())
#endif
{
}

IsoAlignedMemoryAllocator::~IsoAlignedMemoryAllocator()
{
    releaseMemoryFromSubclassDestructor();
}

void IsoAlignedMemoryAllocator::dump(PrintStream& out) const
{
    out.print("Iso(", RawPointer(this), ")");
}

void* IsoAlignedMemoryAllocator::tryAllocateMemory(size_t size)
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    return m_heap.malloc(size);
#else
    return FastCompactMalloc::tryMalloc(size);
#endif
}

void IsoAlignedMemoryAllocator::freeMemory(void* pointer)
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    return m_heap.free(pointer);
#else
    FastMalloc::free(pointer);
#endif
}

void* IsoAlignedMemoryAllocator::tryReallocateMemory(void*, size_t)
{
    // In IsoSubspace-managed PreciseAllocation, we must not perform realloc.
    RELEASE_ASSERT_NOT_REACHED();
}

void* IsoAlignedMemoryAllocator::tryMallocBlock()
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    return m_heap.memalign(MarkedBlock::blockSize, MarkedBlock::blockSize, true);
#else
    return tryFastCompactAlignedMalloc(MarkedBlock::blockSize, MarkedBlock::blockSize);
#endif
}

void IsoAlignedMemoryAllocator::freeBlock(void* block)
{
#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    m_heap.free(block);
#else
    fastAlignedFree(block);
#endif
}

void IsoAlignedMemoryAllocator::commitBlock(void* block)
{
    WTF::fastCommitAlignedMemory(block, MarkedBlock::blockSize);
}

void IsoAlignedMemoryAllocator::decommitBlock(void* block)
{
    WTF::fastDecommitAlignedMemory(block, MarkedBlock::blockSize);
}

} // namespace JSC

