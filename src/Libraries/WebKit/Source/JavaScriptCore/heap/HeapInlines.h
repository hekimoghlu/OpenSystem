/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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

#include "GCDeferralContext.h"
#include "Heap.h"
#include "HeapCellInlines.h"
#include "IndexingHeader.h"
#include "JSCast.h"
#include "Structure.h"
#include <type_traits>
#include <wtf/Assertions.h>
#include <wtf/MainThread.h>

namespace JSC {

ALWAYS_INLINE VM& Heap::vm() const
{
    return *std::bit_cast<VM*>(std::bit_cast<uintptr_t>(this) - OBJECT_OFFSETOF(VM, heap));
}

ALWAYS_INLINE JSC::Heap* Heap::heap(const HeapCell* cell)
{
    if (!cell)
        return nullptr;
    return cell->heap();
}

inline JSC::Heap* Heap::heap(const JSValue v)
{
    if (!v.isCell())
        return nullptr;
    return heap(v.asCell());
}

inline bool Heap::hasHeapAccess() const
{
    return m_worldState.load() & hasAccessBit;
}

inline bool Heap::worldIsStopped() const
{
    return m_worldIsStopped;
}

ALWAYS_INLINE bool Heap::isMarked(const void* rawCell)
{
    ASSERT(!m_isMarkingForGCVerifier);
    HeapCell* cell = std::bit_cast<HeapCell*>(rawCell);
    if (cell->isPreciseAllocation())
        return cell->preciseAllocation().isMarked();
    MarkedBlock& block = cell->markedBlock();
    return block.isMarked(m_objectSpace.markingVersion(), cell);
}

ALWAYS_INLINE bool Heap::testAndSetMarked(HeapVersion markingVersion, const void* rawCell)
{
    HeapCell* cell = std::bit_cast<HeapCell*>(rawCell);
    if (cell->isPreciseAllocation())
        return cell->preciseAllocation().testAndSetMarked();
    MarkedBlock& block = cell->markedBlock();
    Dependency dependency = block.aboutToMark(markingVersion, cell);
    return block.testAndSetMarked(cell, dependency);
}

ALWAYS_INLINE size_t Heap::cellSize(const void* rawCell)
{
    return std::bit_cast<HeapCell*>(rawCell)->cellSize();
}

inline void Heap::writeBarrier(const JSCell* from, JSValue to)
{
#if ENABLE(WRITE_BARRIER_PROFILING)
    WriteBarrierCounters::countWriteBarrier();
#endif
    if (!to.isCell())
        return;
    writeBarrier(from, to.asCell());
}

inline void Heap::writeBarrier(const JSCell* from, JSCell* to)
{
#if ENABLE(WRITE_BARRIER_PROFILING)
    WriteBarrierCounters::countWriteBarrier();
#endif
    if (!from)
        return;
    if (LIKELY(!to))
        return;
    if (!isWithinThreshold(from->cellState(), barrierThreshold()))
        return;
    writeBarrierSlowPath(from);
}

inline void Heap::writeBarrier(const JSCell* from)
{
    ASSERT_GC_OBJECT_LOOKS_VALID(const_cast<JSCell*>(from));
    if (!from)
        return;
    if (UNLIKELY(isWithinThreshold(from->cellState(), barrierThreshold())))
        writeBarrierSlowPath(from);
}

inline void Heap::mutatorFence()
{
    if (isX86() || UNLIKELY(mutatorShouldBeFenced()))
        WTF::storeStoreFence();
}

template<typename Functor> inline void Heap::forEachCodeBlock(const Functor& func)
{
    forEachCodeBlockImpl(scopedLambdaRef<void(CodeBlock*)>(func));
}

template<typename Functor> inline void Heap::forEachCodeBlockIgnoringJITPlans(const AbstractLocker& codeBlockSetLocker, const Functor& func)
{
    forEachCodeBlockIgnoringJITPlansImpl(codeBlockSetLocker, scopedLambdaRef<void(CodeBlock*)>(func));
}

template<typename Functor> inline void Heap::forEachProtectedCell(const Functor& functor)
{
    for (auto& pair : m_protectedValues)
        functor(pair.key);
    m_handleSet.forEachStrongHandle(functor, m_protectedValues);
}

#if USE(FOUNDATION)
template <typename T>
inline void Heap::releaseSoon(RetainPtr<T>&& object)
{
    m_delayedReleaseObjects.append(WTFMove(object));
}
#endif

#ifdef JSC_GLIB_API_ENABLED
inline void Heap::releaseSoon(std::unique_ptr<JSCGLibWrapperObject>&& object)
{
    m_delayedReleaseObjects.append(WTFMove(object));
}
#endif

inline void Heap::incrementDeferralDepth()
{
    ASSERT(!Thread::mayBeGCThread() || m_worldIsStopped);
    m_deferralDepth++;
}

inline void Heap::decrementDeferralDepth()
{
    ASSERT(!Thread::mayBeGCThread() || m_worldIsStopped);
    m_deferralDepth--;
}

inline void Heap::decrementDeferralDepthAndGCIfNeeded()
{
    ASSERT(!Thread::mayBeGCThread() || m_worldIsStopped);
    m_deferralDepth--;
    
    if (UNLIKELY(m_didDeferGCWork) || Options::forceDidDeferGCWork()) {
        decrementDeferralDepthAndGCIfNeededSlow();
        
        // Here are the possible relationships between m_deferralDepth and m_didDeferGCWork.
        // Note that prior to the call to decrementDeferralDepthAndGCIfNeededSlow,
        // m_didDeferGCWork had to have been true. Now it can be either false or true. There is
        // nothing we can reliably assert.
        //
        // Possible arrangements of m_didDeferGCWork and !!m_deferralDepth:
        //
        // Both false: We popped out of all DeferGCs and we did whatever work was deferred.
        //
        // Only m_didDeferGCWork is true: We stopped for GC and the GC did DeferGC. This is
        // possible because of how we handle the baseline JIT's worklist. It's also perfectly
        // safe because it only protects reportExtraMemory. We can just ignore this.
        //
        // Only !!m_deferralDepth is true: m_didDeferGCWork had been set spuriously. It is only
        // cleared by decrementDeferralDepthAndGCIfNeededSlow(). So, if we had deferred work but
        // then decrementDeferralDepth()'d, then we might have the bit set even if we GC'd since
        // then.
        //
        // Both true: We're in a recursive ~DeferGC. We wanted to do something about the
        // deferred work, but were unable to.
    }
}

inline UncheckedKeyHashSet<MarkedVectorBase*>& Heap::markListSet()
{
    return m_markListSet;
}

inline void Heap::reportExtraMemoryAllocated(const JSCell* cell, size_t size)
{
    if (size > minExtraMemory)
        reportExtraMemoryAllocatedSlowCase(nullptr, cell, size);
}

inline void Heap::reportExtraMemoryAllocated(GCDeferralContext* deferralContext, const JSCell* cell, size_t size)
{
    if (size > minExtraMemory)
        reportExtraMemoryAllocatedSlowCase(deferralContext, cell, size);
}

inline void Heap::deprecatedReportExtraMemory(size_t size)
{
    if (size > minExtraMemory) 
        deprecatedReportExtraMemorySlowCase(size);
}

inline void Heap::acquireAccess()
{
    if constexpr (validateDFGDoesGC)
        vm().verifyCanGC();

    if (m_worldState.compareExchangeWeak(0, hasAccessBit))
        return;
    acquireAccessSlow();
}

inline bool Heap::hasAccess() const
{
    return m_worldState.loadRelaxed() & hasAccessBit;
}

inline void Heap::releaseAccess()
{
    if (m_worldState.compareExchangeWeak(hasAccessBit, 0))
        return;
    releaseAccessSlow();
}

inline bool Heap::mayNeedToStop()
{
    return m_worldState.loadRelaxed() != hasAccessBit;
}

inline void Heap::stopIfNecessary()
{
    if constexpr (validateDFGDoesGC)
        vm().verifyCanGC();

    if (mayNeedToStop())
        stopIfNecessarySlow();
}

template<typename Func>
void Heap::forEachSlotVisitor(const Func& func)
{
    func(*m_collectorSlotVisitor);
    func(*m_mutatorSlotVisitor);
    for (auto& visitor : m_parallelSlotVisitors)
        func(*visitor);
}

namespace GCClient {

ALWAYS_INLINE VM& Heap::vm() const
{
    return *std::bit_cast<VM*>(std::bit_cast<uintptr_t>(this) - OBJECT_OFFSETOF(VM, clientHeap));
}

} // namespace GCClient

} // namespace JSC
