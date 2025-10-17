/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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

#include "AbstractSlotVisitor.h"
#include "JSCJSValue.h"
#include "MarkedBlock.h"
#include "VisitRaceKey.h"
#include "Weak.h"
#include <memory>
#include <wtf/BitSet.h>
#include <wtf/Deque.h>
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/SharedTask.h>
#include <wtf/TZoneMalloc.h>

namespace WTF {

class StackTrace;

} // namespace WTF

namespace JSC {

class ConservativeRoots;
class Heap;
class HeapCell;
class PreciseAllocation;

using WTF::StackTrace;

class VerifierSlotVisitor : public AbstractSlotVisitor {
    WTF_MAKE_NONCOPYABLE(VerifierSlotVisitor);
    WTF_MAKE_TZONE_ALLOCATED(VerifierSlotVisitor);
    using Base = AbstractSlotVisitor;
public:
    using ReferrerToken = AbstractSlotVisitor::ReferrerToken;

    struct MarkerData {
        MarkerData() = default;
        MarkerData(MarkerData&&) = default;
        MarkerData(ReferrerToken, std::unique_ptr<StackTrace>&&);
        MarkerData& operator=(MarkerData&&) = default;

        ReferrerToken referrer() const { return m_referrer; }
        StackTrace* stack() const { return m_stack.get(); }

    private:
        ReferrerToken m_referrer;
        std::unique_ptr<StackTrace> m_stack;
    };

    VerifierSlotVisitor(Heap&);
    ~VerifierSlotVisitor();

    void append(const ConservativeRoots&) final;
    void appendUnbarriered(JSCell*) final;
    void appendHiddenUnbarriered(JSCell*) final;

    bool isFirstVisit() const final;
    bool isMarked(const void*) const final;
    bool isMarked(MarkedBlock&, HeapCell*) const final;
    bool isMarked(PreciseAllocation&, HeapCell*) const final;

    void markAuxiliary(const void*) final;

    void reportExtraMemoryVisited(size_t) final { }
#if ENABLE(RESOURCE_USAGE)
    void reportExternalMemoryVisited(size_t) final { }
#endif

    bool mutatorIsStopped() const final;

    void didAddOpaqueRoot(void*) final;
    void didFindOpaqueRoot(void*) final;

    void didRace(const VisitRaceKey&) final { }
    void dump(PrintStream&) const final;

    void visitAsConstraint(const JSCell*) final;

    void drain();

    void addParallelConstraintTask(RefPtr<SharedTask<void(AbstractSlotVisitor&)>>) final;
    NO_RETURN_DUE_TO_CRASH void addParallelConstraintTask(RefPtr<SharedTask<void(SlotVisitor&)>>) final;
    void executeConstraintTasks();

    template<typename Functor> void forEachLiveCell(const Functor&);
    template<typename Functor> void forEachLivePreciseAllocation(const Functor&);
    template<typename Functor> void forEachLiveMarkedBlockCell(const Functor&);

    JS_EXPORT_PRIVATE void dumpMarkerData(HeapCell*);

    bool doneMarking() { return m_doneMarking; }
    void setDoneMarking()
    {
        ASSERT(!m_doneMarking);
        m_doneMarking = true;
    }

private:
    class MarkedBlockData {
        WTF_MAKE_TZONE_ALLOCATED(MarkedBlockData);
        WTF_MAKE_NONCOPYABLE(MarkedBlockData);
    public:
        using AtomsBitSet = WTF::BitSet<MarkedBlock::atomsPerBlock>;

        MarkedBlockData(MarkedBlock*);

        MarkedBlock* block() const { return m_block; }
        const AtomsBitSet& atoms() const { return m_atoms; }

        bool isMarked(unsigned atomNumber) { return m_atoms.get(atomNumber); }
        bool testAndSetMarked(unsigned atomNumber) { return m_atoms.testAndSet(atomNumber); }

        void addMarkerData(unsigned atomNumber, MarkerData&&);
        const MarkerData* markerData(unsigned atomNumber) const;

    private:
        MarkedBlock* m_block { nullptr };
        AtomsBitSet m_atoms;
        Vector<MarkerData> m_markers;
    };

    class PreciseAllocationData {
        WTF_MAKE_TZONE_ALLOCATED(PreciseAllocationData);
        WTF_MAKE_NONCOPYABLE(PreciseAllocationData);
    public:
        PreciseAllocationData(PreciseAllocation*);
        PreciseAllocation* allocation() const { return m_allocation; }

        void addMarkerData(MarkerData&&);
        const MarkerData* markerData() const;

    private:
        PreciseAllocation* m_allocation { nullptr };
        MarkerData m_marker;
    };

    class OpaqueRootData {
        WTF_MAKE_TZONE_ALLOCATED(OpaqueRootData);
        WTF_MAKE_NONCOPYABLE(OpaqueRootData);
    public:
        OpaqueRootData() = default;

        void addMarkerData(MarkerData&&);
        const MarkerData* markerData() const;

    private:
        MarkerData m_marker;
    };

    using MarkedBlockMap = UncheckedKeyHashMap<MarkedBlock*, std::unique_ptr<MarkedBlockData>>;
    using PreciseAllocationMap = UncheckedKeyHashMap<PreciseAllocation*, std::unique_ptr<PreciseAllocationData>>;
    using OpaqueRootMap = UncheckedKeyHashMap<void*, std::unique_ptr<OpaqueRootData>>;

    void appendToMarkStack(JSCell*);
    void appendSlow(JSCell* cell) { setMarkedAndAppendToMarkStack(cell); }

    bool testAndSetMarked(const void* rawCell);
    bool testAndSetMarked(PreciseAllocation&);
    bool testAndSetMarked(MarkedBlock&, HeapCell*);

    void setMarkedAndAppendToMarkStack(JSCell*);

    void visitChildren(const JSCell*);

    OpaqueRootMap m_opaqueRootMap;
    PreciseAllocationMap m_preciseAllocationMap;
    MarkedBlockMap m_markedBlockMap;
    ConcurrentPtrHashSet m_opaqueRootStorage;
    Deque<RefPtr<SharedTask<void(AbstractSlotVisitor&)>>, 32> m_constraintTasks;
    bool m_doneMarking { false };
};

} // namespace JSC
