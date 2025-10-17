/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#pragma once

#include "BlockDirectoryInlines.h"
#include "HeapCellType.h"
#include "JSCast.h"
#include "MarkedBlock.h"
#include "MarkedSpace.h"
#include "Subspace.h"

namespace JSC {

template<typename Func>
void Subspace::forEachDirectory(const Func& func)
{
    for (BlockDirectory* directory = m_firstDirectory; directory; directory = directory->nextDirectoryInSubspace())
        func(*directory);
}

template<typename Func>
void Subspace::forEachMarkedBlock(const Func& func)
{
    forEachDirectory(
        [&] (BlockDirectory& directory) {
            directory.forEachBlock(func);
        });
}

template<typename Func>
void Subspace::forEachNotEmptyMarkedBlock(const Func& func)
{
    forEachDirectory(
        [&] (BlockDirectory& directory) {
            directory.forEachNotEmptyBlock(func);
        });
}

template<typename Func>
void Subspace::forEachPreciseAllocation(const Func& func)
{
    for (PreciseAllocation& allocation : m_preciseAllocations)
        func(&allocation);
}

template<typename Func>
void Subspace::forEachMarkedCell(const Func& func)
{
    forEachNotEmptyMarkedBlock(
        [&] (MarkedBlock::Handle* handle) {
            handle->forEachMarkedCell(
                [&] (size_t, HeapCell* cell, HeapCell::Kind kind) -> IterationStatus {
                    func(cell, kind);
                    return IterationStatus::Continue;
                });
        });
    CellAttributes attributes = this->attributes();
    forEachPreciseAllocation(
        [&] (PreciseAllocation* allocation) {
            if (allocation->isMarked())
                func(allocation->cell(), attributes.cellKind);
        });
}

template<typename Visitor, typename Func>
Ref<SharedTask<void(Visitor&)>> Subspace::forEachMarkedCellInParallel(const Func& func)
{
    class Task final : public SharedTask<void(Visitor&)> {
    public:
        Task(Subspace& subspace, const Func& func)
            : m_subspace(subspace)
            , m_blockSource(subspace.parallelNotEmptyMarkedBlockSource())
            , m_func(func)
        {
        }
        
        void run(Visitor& visitor) final
        {
            while (MarkedBlock::Handle* handle = m_blockSource->run()) {
                handle->forEachMarkedCell(
                    [&] (size_t, HeapCell* cell, HeapCell::Kind kind) -> IterationStatus {
                        m_func(visitor, cell, kind);
                        return IterationStatus::Continue;
                    });
            }

            if (m_doneVisitingPreciseAllocations.test_and_set(std::memory_order_relaxed))
                return;

            CellAttributes attributes = m_subspace.attributes();
            m_subspace.forEachPreciseAllocation(
                [&] (PreciseAllocation* allocation) {
                    if (allocation->isMarked())
                        m_func(visitor, allocation->cell(), attributes.cellKind);
                });
        }
        
    private:
        Subspace& m_subspace;
        Ref<SharedTask<MarkedBlock::Handle*()>> m_blockSource;
        Func m_func;
        std::atomic_flag m_doneVisitingPreciseAllocations { };
    };
    
    return adoptRef(*new Task(*this, func));
}

template<typename Func>
void Subspace::forEachLiveCell(const Func& func)
{
    forEachMarkedBlock(
        [&] (MarkedBlock::Handle* handle) {
            handle->forEachLiveCell(
                [&] (size_t, HeapCell* cell, HeapCell::Kind kind) -> IterationStatus {
                    func(cell, kind);
                    return IterationStatus::Continue;
                });
        });
    CellAttributes attributes = this->attributes();
    forEachPreciseAllocation(
        [&] (PreciseAllocation* allocation) {
            if (allocation->isLive())
                func(allocation->cell(), attributes.cellKind);
        });
}

inline CellAttributes Subspace::attributes() const
{
    return m_heapCellType->attributes();
}

} // namespace JSC

