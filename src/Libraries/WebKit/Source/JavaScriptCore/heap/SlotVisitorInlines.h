/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#include "AbstractSlotVisitorInlines.h"
#include "MarkedBlock.h"
#include "PreciseAllocation.h"
#include "SlotVisitor.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

ALWAYS_INLINE void SlotVisitor::appendUnbarriered(JSValue* slot, size_t count)
{
    for (size_t i = count; i--;)
        appendUnbarriered(slot[i]);
}

ALWAYS_INLINE void SlotVisitor::appendUnbarriered(JSCell* cell)
{
    // This needs to be written in a very specific way to ensure that it gets inlined
    // properly. In particular, it appears that using templates here breaks ALWAYS_INLINE.
    
    if (!cell)
        return;
    
    Dependency dependency;
    if (UNLIKELY(cell->isPreciseAllocation())) {
        if (LIKELY(cell->preciseAllocation().isMarked())) {
            if (LIKELY(!m_heapAnalyzer))
                return;
        }
    } else {
        MarkedBlock& block = cell->markedBlock();
        dependency = block.aboutToMark(m_markingVersion, cell);
        if (LIKELY(block.isMarked(cell, dependency))) {
            if (LIKELY(!m_heapAnalyzer))
                return;
        }
    }
    
    appendSlow(cell, dependency);
}

ALWAYS_INLINE void SlotVisitor::appendUnbarriered(JSValue value)
{
    if (value.isCell())
        appendUnbarriered(value.asCell());
}

ALWAYS_INLINE void SlotVisitor::appendHiddenUnbarriered(JSValue value)
{
    if (value.isCell())
        appendHiddenUnbarriered(value.asCell());
}

ALWAYS_INLINE void SlotVisitor::appendHiddenUnbarriered(JSCell* cell)
{
    // This needs to be written in a very specific way to ensure that it gets inlined
    // properly. In particular, it appears that using templates here breaks ALWAYS_INLINE.
    
    if (!cell)
        return;
    
    Dependency dependency;
    if (UNLIKELY(cell->isPreciseAllocation())) {
        if (LIKELY(cell->preciseAllocation().isMarked()))
            return;
    } else {
        MarkedBlock& block = cell->markedBlock();
        dependency = block.aboutToMark(m_markingVersion, cell);
        if (LIKELY(block.isMarked(cell, dependency)))
            return;
    }
    
    appendHiddenSlow(cell, dependency);
}

template<typename T>
ALWAYS_INLINE void SlotVisitor::append(const Weak<T>& weak)
{
    appendUnbarriered(weak.get());
}

template<typename T, typename Traits>
ALWAYS_INLINE void SlotVisitor::append(const WriteBarrierBase<T, Traits>& slot)
{
    appendUnbarriered(slot.get());
}

template<typename T, typename Traits>
ALWAYS_INLINE void SlotVisitor::appendHidden(const WriteBarrierBase<T, Traits>& slot)
{
    appendHiddenUnbarriered(slot.get());
}

ALWAYS_INLINE void SlotVisitor::append(const WriteBarrierStructureID& slot)
{
    appendUnbarriered(reinterpret_cast<JSCell*>(slot.get()));
}

ALWAYS_INLINE void SlotVisitor::appendHidden(const WriteBarrierStructureID& slot)
{
    appendHiddenUnbarriered(reinterpret_cast<JSCell*>(slot.get()));
}

template<typename Iterator>
ALWAYS_INLINE void SlotVisitor::append(Iterator begin, Iterator end)
{
    for (auto it = begin; it != end; ++it)
        append(*it);
}

ALWAYS_INLINE void SlotVisitor::appendValues(std::span<const WriteBarrier<Unknown>> barriers)
{
    for (auto& barrier : barriers)
        append(barrier);
}

ALWAYS_INLINE void SlotVisitor::appendValues(const WriteBarrierBase<Unknown>* barriers, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        append(barriers[i]);
}

ALWAYS_INLINE void SlotVisitor::appendValuesHidden(const WriteBarrierBase<Unknown>* barriers, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        appendHidden(barriers[i]);
}

ALWAYS_INLINE bool SlotVisitor::isMarked(const void* p) const
{
    return heap()->isMarked(p);
}

ALWAYS_INLINE bool SlotVisitor::isMarked(MarkedBlock& container, HeapCell* cell) const
{
    return container.isMarked(markingVersion(), cell);
}

ALWAYS_INLINE bool SlotVisitor::isMarked(PreciseAllocation& container, HeapCell* cell) const
{
    return container.isMarked(markingVersion(), cell);
}

inline void SlotVisitor::reportExtraMemoryVisited(size_t size)
{
    if (m_isFirstVisit) {
        m_nonCellVisitCount += size;
        // FIXME: Change this to use SaturatedArithmetic when available.
        // https://bugs.webkit.org/show_bug.cgi?id=170411
        m_extraMemorySize += size;
    }
}

#if ENABLE(RESOURCE_USAGE)
inline void SlotVisitor::reportExternalMemoryVisited(size_t size)
{
    if (m_isFirstVisit)
        heap()->reportExternalMemoryVisited(size);
}
#endif

template<typename Func>
IterationStatus SlotVisitor::forEachMarkStack(const Func& func)
{
    if (func(m_collectorStack) == IterationStatus::Done)
        return IterationStatus::Done;
    if (func(m_mutatorStack) == IterationStatus::Done)
        return IterationStatus::Done;
    return IterationStatus::Continue;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
