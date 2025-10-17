/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include "IsoMemoryAllocatorBase.h"

#include "MarkedBlock.h"
#include <wtf/text/CString.h>

namespace JSC {

IsoMemoryAllocatorBase::IsoMemoryAllocatorBase(CString name)
{
    UNUSED_PARAM(name);
}

IsoMemoryAllocatorBase::~IsoMemoryAllocatorBase() = default;

// We need to call this from the derived class's destructor because it's undefined behavior to call pure virtual methods from within a destructor.
void IsoMemoryAllocatorBase::releaseMemoryFromSubclassDestructor()
{
#if !ENABLE(MALLOC_HEAP_BREAKDOWN)
    for (unsigned i = 0; i < m_blocks.size(); ++i) {
        void* block = m_blocks[i];
        if (!m_committed.quickGet(i))
            commitBlock(block);
        freeBlock(block);
    }
#endif
}

void* IsoMemoryAllocatorBase::tryAllocateAlignedMemory(size_t alignment, size_t size)
{
    // Since this is designed specially for IsoSubspace, we know that we will only be asked to
    // allocate MarkedBlocks.
    RELEASE_ASSERT(alignment == MarkedBlock::blockSize);
    RELEASE_ASSERT(size == MarkedBlock::blockSize);

    Locker locker { m_lock };
    
    m_firstUncommitted = m_committed.findBit(m_firstUncommitted, false);
    if (m_firstUncommitted < m_blocks.size()) {
        m_committed.quickSet(m_firstUncommitted);
        void* result = m_blocks[m_firstUncommitted];
        commitBlock(result);
        return result;
    }
    
    void* result = tryMallocBlock();
    if (!result)
        return nullptr;
    unsigned index = m_blocks.size();
    m_blocks.append(result);
    m_blockIndices.add(result, index);
    if (m_blocks.capacity() != m_committed.size())
        m_committed.resize(m_blocks.capacity());
    m_committed.quickSet(index);
    return result;
}

void IsoMemoryAllocatorBase::freeAlignedMemory(void* basePtr)
{
    Locker locker { m_lock };
    
    auto iter = m_blockIndices.find(basePtr);
    RELEASE_ASSERT(iter != m_blockIndices.end());
    unsigned index = iter->value;
    m_committed.quickClear(index);
    m_firstUncommitted = std::min(index, m_firstUncommitted);
    decommitBlock(basePtr);
}

} // namespace JSC

