/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include "MarkedBlockInlines.h"
#include "MarkedSpace.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

ALWAYS_INLINE JSC::Heap& MarkedSpace::heap() const
{
    return *std::bit_cast<Heap*>(std::bit_cast<uintptr_t>(this) - OBJECT_OFFSETOF(Heap, m_objectSpace));
}

template<typename Functor> inline void MarkedSpace::forEachLiveCell(HeapIterationScope&, const Functor& functor)
{
    ASSERT(isIterating());
    forEachLiveCell(functor);
}

template<typename Functor> inline void MarkedSpace::forEachLiveCell(const Functor& functor)
{
    BlockIterator end = m_blocks.set().end();
    for (BlockIterator it = m_blocks.set().begin(); it != end; ++it) {
        IterationStatus result = (*it)->handle().forEachLiveCell(
            [&] (size_t, HeapCell* cell, HeapCell::Kind kind) -> IterationStatus {
                return functor(cell, kind);
            });
        if (result == IterationStatus::Done)
            return;
    }
    for (PreciseAllocation* allocation : m_preciseAllocations) {
        if (allocation->isLive()) {
            if (functor(allocation->cell(), allocation->attributes().cellKind) == IterationStatus::Done)
                return;
        }
    }
}

template<typename Functor> inline void MarkedSpace::forEachDeadCell(HeapIterationScope&, const Functor& functor)
{
    ASSERT(isIterating());
    BlockIterator end = m_blocks.set().end();
    for (BlockIterator it = m_blocks.set().begin(); it != end; ++it) {
        if ((*it)->handle().forEachDeadCell(functor) == IterationStatus::Done)
            return;
    }
    for (PreciseAllocation* allocation : m_preciseAllocations) {
        if (!allocation->isLive()) {
            if (functor(allocation->cell(), allocation->attributes().cellKind) == IterationStatus::Done)
                return;
        }
    }
}

template<typename Visitor>
inline Ref<SharedTask<void(Visitor&)>> MarkedSpace::forEachWeakInParallel(Visitor& visitor)
{
    constexpr unsigned batchSize = 16;
    class Task final : public SharedTask<void(Visitor&)> {
    public:
        Task(MarkedSpace& markedSpace, Visitor& visitor)
            : m_markedSpace(markedSpace)
            , m_newActiveCursor(markedSpace.m_newActiveWeakSets.begin())
            , m_activeCursor(markedSpace.heap().collectionScope() == CollectionScope::Full ? markedSpace.m_activeWeakSets.begin() : markedSpace.m_activeWeakSets.end())
            , m_reason(visitor.rootMarkReason())
        {
        }

        std::span<WeakBlock*> drain(std::array<WeakBlock*, batchSize>& results)
        {
            Locker locker { m_lock };
            size_t resultsSize = 0;
            while (true) {
                if (m_current) {
                    auto* block = m_current;
                    m_current = m_current->next();
                    if (block->isEmpty())
                        continue;
                    results[resultsSize++] = block;
                    if (resultsSize == batchSize)
                        return std::span { results.data(), resultsSize };
                    continue;
                }

                if (m_newActiveCursor != m_markedSpace.m_newActiveWeakSets.end()) {
                    m_current = m_newActiveCursor->head();
                    ++m_newActiveCursor;
                    continue;
                }

                if (m_activeCursor != m_markedSpace.m_activeWeakSets.end()) {
                    m_current = m_activeCursor->head();
                    ++m_activeCursor;
                    continue;
                }
                return std::span { results.data(), resultsSize };
            }
        }

        void run(Visitor& visitor) final
        {
            SetRootMarkReasonScope rootScope(visitor, m_reason);
            std::array<WeakBlock*, batchSize> resultsStorage;
            while (true) {
                auto results = drain(resultsStorage);
                if (!results.size())
                    return;
                for (auto* result : results)
                    result->visit(visitor);
            }
        }

    private:
        MarkedSpace& m_markedSpace;
        WeakBlock* m_current { nullptr };
        SentinelLinkedList<WeakSet, BasicRawSentinelNode<WeakSet>>::iterator m_newActiveCursor;
        SentinelLinkedList<WeakSet, BasicRawSentinelNode<WeakSet>>::iterator m_activeCursor;
        Lock m_lock;
        RootMarkReason m_reason;
    };

    return adoptRef(*new Task(*this, visitor));
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
