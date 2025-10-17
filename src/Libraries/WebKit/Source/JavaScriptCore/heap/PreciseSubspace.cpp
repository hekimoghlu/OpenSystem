/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#include "PreciseSubspace.h"

#include "AllocatingScope.h"
#include "IsoAlignedMemoryAllocator.h"
#include "IsoCellSetInlines.h"
#include "JSCellInlines.h"
#include "MarkedSpaceInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PreciseSubspace);

PreciseSubspace::PreciseSubspace(CString name, JSC::Heap& heap, const HeapCellType& heapCellType, AlignedMemoryAllocator* allocator)
    : Subspace(name, heap)
{
    m_isPreciseOnly = true;
    initialize(heapCellType, allocator);
}

PreciseSubspace::~PreciseSubspace() = default;

void PreciseSubspace::didResizeBits(unsigned)
{
    UNREACHABLE_FOR_PLATFORM();
}

void PreciseSubspace::didRemoveBlock(unsigned)
{
    UNREACHABLE_FOR_PLATFORM();
}

void PreciseSubspace::didBeginSweepingToFreeList(MarkedBlock::Handle*)
{
    UNREACHABLE_FOR_PLATFORM();
}

void* PreciseSubspace::tryAllocate(size_t size)
{
    PreciseAllocation* allocation = PreciseAllocation::tryCreate(m_space.heap(), size, this, 0);
    if (!allocation)
        return nullptr;

    m_preciseAllocations.append(allocation);
    m_space.registerPreciseAllocation(allocation, /* isNewAllocation */ true);
    return allocation->cell();
}

void* PreciseSubspace::allocate(VM& vm, size_t cellSize, GCDeferralContext* deferralContext, AllocationFailureMode failureMode)
{
    ASSERT(vm.currentThreadIsHoldingAPILock());
    ASSERT(!vm.heap.objectSpace().isIterating());
    UNUSED_PARAM(failureMode);
    ASSERT_WITH_MESSAGE(failureMode == AllocationFailureMode::ReturnNull, "PreciseSubspace is meant for large objects so it should always be able to fail.");

    AllocatingScope helpingHeap(vm.heap);

    vm.heap.collectIfNecessaryOrDefer(deferralContext);

    return tryAllocate(cellSize);
}

} // namespace JSC

