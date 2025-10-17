/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
#include "HeapCellType.h"
#include "MarkedSpaceInlines.h"
#include "ParallelSourceAdapter.h"
#include "SubspaceInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Subspace);

Subspace::Subspace(CString name, JSC::Heap& heap)
    : m_space(heap.objectSpace())
    , m_name(name)
{
}

void Subspace::initialize(const HeapCellType& heapCellType, AlignedMemoryAllocator* alignedMemoryAllocator)
{
    m_heapCellType = &heapCellType;
    m_alignedMemoryAllocator = alignedMemoryAllocator;
    m_directoryForEmptyAllocation = m_alignedMemoryAllocator->firstDirectory();

    JSC::Heap& heap = m_space.heap();
    heap.objectSpace().m_subspaces.append(this);
    m_alignedMemoryAllocator->registerSubspace(this);
}

Subspace::~Subspace() = default;

void Subspace::finishSweep(MarkedBlock::Handle& block, FreeList* freeList)
{
    m_heapCellType->finishSweep(block, freeList);
}

void Subspace::destroy(VM& vm, JSCell* cell)
{
    m_heapCellType->destroy(vm, cell);
}

void Subspace::prepareForAllocation()
{
    forEachDirectory(
        [&] (BlockDirectory& directory) {
            directory.prepareForAllocation();
        });

    m_directoryForEmptyAllocation = m_alignedMemoryAllocator->firstDirectory();
}

MarkedBlock::Handle* Subspace::findEmptyBlockToSteal()
{
    for (; m_directoryForEmptyAllocation; m_directoryForEmptyAllocation = m_directoryForEmptyAllocation->nextDirectoryInAlignedMemoryAllocator()) {
        if (MarkedBlock::Handle* block = m_directoryForEmptyAllocation->findEmptyBlockToSteal())
            return block;
    }
    return nullptr;
}

Ref<SharedTask<BlockDirectory*()>> Subspace::parallelDirectorySource()
{
    class Task final : public SharedTask<BlockDirectory*()> {
    public:
        Task(BlockDirectory* directory)
            : m_directory(directory)
        {
        }
        
        BlockDirectory* run() final
        {
            Locker locker { m_lock };
            BlockDirectory* result = m_directory;
            if (result)
                m_directory = result->nextDirectoryInSubspace();
            return result;
        }
        
    private:
        BlockDirectory* m_directory;
        Lock m_lock;
    };
    
    return adoptRef(*new Task(m_firstDirectory));
}

Ref<SharedTask<MarkedBlock::Handle*()>> Subspace::parallelNotEmptyMarkedBlockSource()
{
    return createParallelSourceAdapter<BlockDirectory*, MarkedBlock::Handle*>(
        parallelDirectorySource(),
        [] (BlockDirectory* directory) -> RefPtr<SharedTask<MarkedBlock::Handle*()>> {
            if (!directory)
                return nullptr;
            return directory->parallelNotEmptyBlockSource();
        });
}

void Subspace::sweepBlocks()
{
    forEachDirectory(
        [&] (BlockDirectory& directory) {
            directory.sweep();
        });
}

void Subspace::didResizeBits(unsigned)
{
}

void Subspace::didRemoveBlock(unsigned)
{
}

void Subspace::didBeginSweepingToFreeList(MarkedBlock::Handle*)
{
}

} // namespace JSC

