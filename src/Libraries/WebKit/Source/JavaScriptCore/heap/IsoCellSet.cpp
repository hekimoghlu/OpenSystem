/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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
#include "IsoCellSet.h"

#include "MarkedBlockInlines.h"

namespace JSC {

IsoCellSet::IsoCellSet(IsoSubspace& subspace)
    : m_subspace(subspace)
{
    ASSERT_WITH_MESSAGE(!subspace.isPreciseOnly(), "IsoSubspaces with precise-only allocations are not supported by IsoCellSet");
    size_t size = subspace.m_directory.m_blocks.size();
    m_blocksWithBits.resize(size);
    m_bits.grow(size);
    subspace.m_cellSets.append(this);
}

IsoCellSet::~IsoCellSet()
{
    if (isOnList())
        BasicRawSentinelNode<IsoCellSet>::remove();
}

Ref<SharedTask<MarkedBlock::Handle*()>> IsoCellSet::parallelNotEmptyMarkedBlockSource()
{
    class Task final : public SharedTask<MarkedBlock::Handle*()> {
    public:
        Task(IsoCellSet& set)
            : m_set(set)
            , m_directory(set.m_subspace.m_directory)
        {
        }
        
        MarkedBlock::Handle* run() final
        {
            if (m_done)
                return nullptr;
            m_directory.assertIsMutatorOrMutatorIsStopped();
            Locker locker { m_lock };
            auto bits = m_directory.markingNotEmptyBitsView() & m_set.m_blocksWithBits;
            m_index = bits.findBit(m_index, true);
            if (m_index >= m_directory.m_blocks.size()) {
                m_done = true;
                return nullptr;
            }
            return m_directory.m_blocks[m_index++];
        }
        
    private:
        IsoCellSet& m_set;
        BlockDirectory& m_directory;
        size_t m_index { 0 };
        Lock m_lock;
        bool m_done { false };
    };
    
    return adoptRef(*new Task(*this));
}

NEVER_INLINE WTF::BitSet<MarkedBlock::atomsPerBlock>* IsoCellSet::addSlow(unsigned blockIndex)
{
    Locker locker { m_subspace.m_directory.m_bitvectorLock };
    auto& bitsPtrRef = m_bits[blockIndex];
    auto* bits = bitsPtrRef.get();
    if (!bits) {
        bitsPtrRef = makeUnique<WTF::BitSet<MarkedBlock::atomsPerBlock>>();
        bits = bitsPtrRef.get();
        WTF::storeStoreFence();
        m_blocksWithBits[blockIndex] = true;
    }
    return bits;
}

void IsoCellSet::didResizeBits(unsigned newSize)
{
    m_blocksWithBits.resize(newSize);
    m_bits.grow(newSize);
}

void IsoCellSet::didRemoveBlock(unsigned blockIndex)
{
    {
        Locker locker { m_subspace.m_directory.m_bitvectorLock };
        m_blocksWithBits[blockIndex] = false;
    }
    m_bits[blockIndex] = nullptr;
}

void IsoCellSet::sweepToFreeList(MarkedBlock::Handle* block)
{
    RELEASE_ASSERT(!block->isAllocated());
    
    if (!m_blocksWithBits[block->index()])
        return;
    
    WTF::loadLoadFence();
    
    if (!m_bits[block->index()]) {
        dataLog("FATAL: for block index ", block->index(), ":\n");
        dataLog("Blocks with bits says: ", !!m_blocksWithBits[block->index()], "\n");
        dataLog("Bits says: ", RawPointer(m_bits[block->index()].get()), "\n");
        RELEASE_ASSERT_NOT_REACHED();
    }
    
    if (block->block().hasAnyNewlyAllocated()) {
        // The newlyAllocated() bits are a superset of the marks() bits.
        m_bits[block->index()]->concurrentFilter(block->block().newlyAllocated());
        return;
    }

    if (block->isEmpty() || block->areMarksStaleForSweep()) {
        {
            // Holding the bitvector lock happens to be enough because that's what we also hold in
            // other places where we manipulate this bitvector.
            Locker locker { m_subspace.m_directory.m_bitvectorLock };
            m_blocksWithBits[block->index()] = false;
        }
        m_bits[block->index()] = nullptr;
        return;
    }
    
    m_bits[block->index()]->concurrentFilter(block->block().marks());
}

} // namespace JSC

