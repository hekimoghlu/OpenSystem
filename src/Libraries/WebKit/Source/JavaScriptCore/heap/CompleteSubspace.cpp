/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#include "Subspace.h"

#include "AlignedMemoryAllocator.h"
#include "AllocatorInlines.h"
#include "JSCellInlines.h"
#include "LocalAllocatorInlines.h"
#include "MarkedSpaceInlines.h"
#include "SubspaceInlines.h"
#include <wtf/RAMSize.h>

namespace JSC {

CompleteSubspace::CompleteSubspace(CString name, JSC::Heap& heap, const HeapCellType& heapCellType, AlignedMemoryAllocator* alignedMemoryAllocator)
    : Subspace(name, heap)
{
    initialize(heapCellType, alignedMemoryAllocator);
}

CompleteSubspace::~CompleteSubspace() = default;

Allocator CompleteSubspace::allocatorForNonInline(size_t size, AllocatorForMode mode)
{
    return allocatorFor(size, mode);
}

Allocator CompleteSubspace::allocatorForSlow(size_t size)
{
    size_t index = MarkedSpace::sizeClassToIndex(size);
    size_t sizeClass = MarkedSpace::s_sizeClassForSizeStep[index];
    if (!sizeClass)
        return Allocator();
    
    // This is written in such a way that it's OK for the JIT threads to end up here if they want
    // to generate code that uses some allocator that hadn't been used yet. Note that a possibly-
    // just-as-good solution would be to return null if we're in the JIT since the JIT treats null
    // allocator as "please always take the slow path". But, that could lead to performance
    // surprises and the algorithm here is pretty easy. Only this code has to hold the lock, to
    // prevent simultaneously BlockDirectory creations from multiple threads. This code ensures
    // that any "forEachAllocator" traversals will only see this allocator after it's initialized
    // enough: it will have 
    Locker locker { m_space.directoryLock() };
    if (Allocator allocator = m_allocatorForSizeStep[index])
        return allocator;

    if (false)
        dataLog("Creating BlockDirectory/LocalAllocator for ", m_name, ", ", attributes(), ", ", sizeClass, ".\n");
    
    std::unique_ptr<BlockDirectory> uniqueDirectory = makeUnique<BlockDirectory>(sizeClass);
    BlockDirectory* directory = uniqueDirectory.get();
    m_directories.append(WTFMove(uniqueDirectory));
    
    directory->setSubspace(this);
    m_space.addBlockDirectory(locker, directory);
    
    std::unique_ptr<LocalAllocator> uniqueLocalAllocator =
        makeUnique<LocalAllocator>(directory);
    LocalAllocator* localAllocator = uniqueLocalAllocator.get();
    m_localAllocators.append(WTFMove(uniqueLocalAllocator));
    
    Allocator allocator(localAllocator);
    
    index = MarkedSpace::sizeClassToIndex(sizeClass);
    for (;;) {
        if (MarkedSpace::s_sizeClassForSizeStep[index] != sizeClass)
            break;

        m_allocatorForSizeStep[index] = allocator;
        
        if (!index--)
            break;
    }
    
    directory->setNextDirectoryInSubspace(m_firstDirectory);
    m_alignedMemoryAllocator->registerDirectory(m_space.heap(), directory);
    WTF::storeStoreFence();
    m_firstDirectory = directory;
    return allocator;
}

void* CompleteSubspace::allocateSlow(VM& vm, size_t size, GCDeferralContext* deferralContext, AllocationFailureMode failureMode)
{
    void* result = tryAllocateSlow(vm, size, deferralContext);
    if (failureMode == AllocationFailureMode::Assert)
        RELEASE_ASSERT(result);
    return result;
}

void* CompleteSubspace::tryAllocateSlow(VM& vm, size_t size, GCDeferralContext* deferralContext)
{
    if constexpr (validateDFGDoesGC)
        vm.verifyCanGC();

    sanitizeStackForVM(vm);

    if (Allocator allocator = allocatorForNonInline(size, AllocatorForMode::EnsureAllocator))
        return allocator.allocate(vm.heap, allocator.cellSize(), deferralContext, AllocationFailureMode::ReturnNull);
    
    if (size <= Options::preciseAllocationCutoff()
        && size <= MarkedSpace::largeCutoff) {
        dataLog("FATAL: attampting to allocate small object using large allocation.\n");
        dataLog("Requested allocation size: ", size, "\n");
        RELEASE_ASSERT_NOT_REACHED();
    }
    
    vm.heap.collectIfNecessaryOrDefer(deferralContext);
    if (UNLIKELY(Options::maxHeapSizeAsRAMSizeMultiple())) {
        if (vm.heap.capacity() > Options::maxHeapSizeAsRAMSizeMultiple() * WTF::ramSize())
            return nullptr;
    }

    size = WTF::roundUpToMultipleOf<MarkedSpace::sizeStep>(size);
    PreciseAllocation* allocation = PreciseAllocation::tryCreate(vm.heap, size, this, m_space.m_preciseAllocations.size());
    if (!allocation)
        return nullptr;
    
    m_preciseAllocations.append(allocation);
    m_space.registerPreciseAllocation(allocation, /* isNewAllocation */ true);
    return allocation->cell();
}

void* CompleteSubspace::reallocatePreciseAllocationNonVirtual(VM& vm, HeapCell* oldCell, size_t size, GCDeferralContext* deferralContext, AllocationFailureMode failureMode)
{
    if constexpr (validateDFGDoesGC)
        vm.verifyCanGC();

    // The following conditions are met in Butterfly for example.
    ASSERT(oldCell->isPreciseAllocation());

    PreciseAllocation* oldAllocation = &oldCell->preciseAllocation();
    ASSERT(oldAllocation->cellSize() <= size);
    ASSERT(oldAllocation->weakSet().isTriviallyDestructible());
    ASSERT(oldAllocation->attributes().destruction == DoesNotNeedDestruction);
    ASSERT(oldAllocation->attributes().cellKind == HeapCell::Auxiliary);
    ASSERT(size > MarkedSpace::largeCutoff);

    sanitizeStackForVM(vm);

    if (size <= Options::preciseAllocationCutoff()
        && size <= MarkedSpace::largeCutoff) {
        dataLog("FATAL: attampting to allocate small object using large allocation.\n");
        dataLog("Requested allocation size: ", size, "\n");
        RELEASE_ASSERT_NOT_REACHED();
    }

    vm.heap.collectIfNecessaryOrDefer(deferralContext);

    size = WTF::roundUpToMultipleOf<MarkedSpace::sizeStep>(size);
    size_t difference = size - oldAllocation->cellSize();
    unsigned oldIndexInSpace = oldAllocation->indexInSpace();
    if (oldAllocation->isOnList())
        oldAllocation->remove();

    PreciseAllocation* allocation = oldAllocation->tryReallocate(size, this);
    if (!allocation) {
        RELEASE_ASSERT(failureMode != AllocationFailureMode::Assert);
        m_preciseAllocations.append(oldAllocation);
        return nullptr;
    }
    ASSERT(oldIndexInSpace == allocation->indexInSpace());

    // If reallocation changes the address, we should update HashSet.
    if (oldAllocation != allocation) {
        if (auto* set = m_space.preciseAllocationSet()) {
            set->remove(oldAllocation->cell());
            set->add(allocation->cell());
        }
    }

    m_space.m_preciseAllocations[oldIndexInSpace] = allocation;
    vm.heap.didAllocate(difference);
    m_space.m_capacity += difference;

    m_preciseAllocations.append(allocation);

    return allocation->cell();
}

} // namespace JSC

