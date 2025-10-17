/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#include "IsoSubspace.h"

#include "IsoAlignedMemoryAllocator.h"
#include "IsoCellSetInlines.h"
#include "JSCellInlines.h"
#include "MarkedSpaceInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IsoSubspace);

IsoSubspace::IsoSubspace(CString name, JSC::Heap& heap, const HeapCellType& heapCellType, size_t size, bool preciseOnly, uint8_t numberOfLowerTierPreciseCells, std::unique_ptr<IsoMemoryAllocatorBase>&& allocator)
    : Subspace(name, heap)
    , m_directory(WTF::roundUpToMultipleOf<MarkedBlock::atomSize>(size))
    , m_isoAlignedMemoryAllocator(allocator ? WTFMove(allocator) : makeUnique<IsoAlignedMemoryAllocator>(name))
{
    if (preciseOnly)
        m_isPreciseOnly = true;
    else {
        m_remainingLowerTierPreciseCount = numberOfLowerTierPreciseCells;
        ASSERT(WTF::roundUpToMultipleOf<MarkedBlock::atomSize>(size) == cellSize());
        ASSERT(m_remainingLowerTierPreciseCount <= MarkedBlock::maxNumberOfLowerTierPreciseCells);
    }

    m_isIsoSubspace = true;
    initialize(heapCellType, m_isoAlignedMemoryAllocator.get());

    Locker locker { m_space.directoryLock() };
    m_directory.setSubspace(this);
    m_space.addBlockDirectory(locker, &m_directory);
    m_alignedMemoryAllocator->registerDirectory(heap, &m_directory);
    m_firstDirectory = &m_directory;
}

IsoSubspace::~IsoSubspace() = default;

void IsoSubspace::didResizeBits(unsigned blockIndex)
{
    m_cellSets.forEach(
        [&] (IsoCellSet* set) {
            set->didResizeBits(blockIndex);
        });
}

void IsoSubspace::didRemoveBlock(unsigned blockIndex)
{
    m_cellSets.forEach(
        [&] (IsoCellSet* set) {
            set->didRemoveBlock(blockIndex);
        });
}

void IsoSubspace::didBeginSweepingToFreeList(MarkedBlock::Handle* block)
{
    m_cellSets.forEach(
        [&] (IsoCellSet* set) {
            set->sweepToFreeList(block);
        });
}

void* IsoSubspace::tryAllocatePreciseOrLowerTierPrecise(size_t size)
{
    auto revive = [&] (PreciseAllocation* allocation) {
        // Lower-tier cells never report capacity. This is intentional since it will not be freed until VM dies.
        // Whether we will do GC or not does not affect on the used memory by lower-tier cells. So we should not
        // count them in capacity since it is not interesting to decide whether we should do GC.
        m_preciseAllocations.append(allocation);
        m_space.registerPreciseAllocation(allocation, /* isNewAllocation */ false);
        ASSERT(allocation->indexInSpace() == m_space.m_preciseAllocations.size() - 1);
        return allocation->cell();
    };

    if (UNLIKELY(m_isPreciseOnly)) {
        PreciseAllocation* allocation = PreciseAllocation::tryCreate(m_space.heap(), size, this, 0);
        return allocation ? revive(allocation) : nullptr;
    }

    ASSERT_WITH_MESSAGE(cellSize() == size, "non-preciseOnly IsoSubspaces shouldn't have variable size");
    if (!m_lowerTierPreciseFreeList.isEmpty()) {
        PreciseAllocation* allocation = &*m_lowerTierPreciseFreeList.begin();
        allocation->remove();
        return revive(allocation);
    }
    if (m_remainingLowerTierPreciseCount) {
        PreciseAllocation* allocation = PreciseAllocation::tryCreateForLowerTierPrecise(m_space.heap(), size, this, --m_remainingLowerTierPreciseCount);
        if (allocation)
            return revive(allocation);
    }
    return nullptr;
}

void IsoSubspace::sweepLowerTierPreciseCell(PreciseAllocation* preciseAllocation)
{
    ASSERT(!m_isPreciseOnly);
    preciseAllocation = preciseAllocation->reuseForLowerTierPrecise();
    m_lowerTierPreciseFreeList.append(preciseAllocation);
}

void IsoSubspace::destroyLowerTierPreciseFreeList()
{
    m_lowerTierPreciseFreeList.forEach([&](PreciseAllocation* allocation) {
        allocation->destroy();
    });
}

namespace GCClient {

IsoSubspace::IsoSubspace(JSC::IsoSubspace& server)
    : m_localAllocator(&server.m_directory)
{
}

} // namespace GCClient

} // namespace JSC

