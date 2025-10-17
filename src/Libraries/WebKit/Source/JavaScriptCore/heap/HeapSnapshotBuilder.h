/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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

#include "HeapAnalyzer.h"
#include <functional>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/OverflowPolicy.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC {

class ConservativeRoots;
class HeapProfiler;
class HeapSnapshot;
class JSCell;

typedef unsigned NodeIdentifier;

struct HeapSnapshotNode {
    HeapSnapshotNode(JSCell* cell, unsigned identifier)
        : cell(cell)
        , identifier(identifier)
    { }

    JSCell* cell;
    NodeIdentifier identifier;
};

enum class EdgeType : uint8_t {
    Internal,     // Normal strong reference. No name.
    Property,     // Named property. In `object.property` the name is "property"
    Index,        // Indexed property. In `array[0]` name is index "0".
    Variable,     // Variable held by a scope. In `let x, f=() => x++` name is "x" in f's captured scope.
    // FIXME: <https://webkit.org/b/154934> Heap Snapshot should include "Weak" edges
};

struct HeapSnapshotEdge {
    HeapSnapshotEdge(JSCell* fromCell, JSCell* toCell)
        : type(EdgeType::Internal)
    {
        from.cell = fromCell;
        to.cell = toCell;
    }

    HeapSnapshotEdge(JSCell* fromCell, JSCell* toCell, EdgeType type, UniquedStringImpl* name)
        : type(type)
    {
        ASSERT(type == EdgeType::Property || type == EdgeType::Variable);
        from.cell = fromCell;
        to.cell = toCell;
        u.name = name;
    }

    HeapSnapshotEdge(JSCell* fromCell, JSCell* toCell, uint32_t index)
        : type(EdgeType::Index)
    {
        from.cell = fromCell;
        to.cell = toCell;
        u.index = index;
    }

    union {
        JSCell *cell;
        NodeIdentifier identifier;
    } from;

    union {
        JSCell *cell;
        NodeIdentifier identifier;
    } to;

    union {
        UniquedStringImpl* name;
        uint32_t index;
    } u;

    EdgeType type;
};

class JS_EXPORT_PRIVATE HeapSnapshotBuilder final : public HeapAnalyzer {
    WTF_MAKE_TZONE_ALLOCATED(HeapSnapshotBuilder);
public:
    enum SnapshotType { InspectorSnapshot, GCDebuggingSnapshot };

    HeapSnapshotBuilder(HeapProfiler&, SnapshotType = SnapshotType::InspectorSnapshot, OverflowPolicy = OverflowPolicy::CrashOnOverflow);
    ~HeapSnapshotBuilder() final;

    static void resetNextAvailableObjectIdentifier();

    // Performs a garbage collection that builds a snapshot of all live cells.
    void buildSnapshot();

    // A root or marked cell.
    void analyzeNode(JSCell*) final;

    // A reference from one cell to another.
    void analyzeEdge(JSCell* from, JSCell* to, RootMarkReason) final;
    void analyzePropertyNameEdge(JSCell* from, JSCell* to, UniquedStringImpl* propertyName) final;
    void analyzeVariableNameEdge(JSCell* from, JSCell* to, UniquedStringImpl* variableName) final;
    void analyzeIndexEdge(JSCell* from, JSCell* to, uint32_t index) final;

    void setOpaqueRootReachabilityReasonForCell(JSCell*, ASCIILiteral) final;
    void setWrappedObjectForCell(JSCell*, void*) final;
    void setLabelForCell(JSCell*, const String&) final;

    String json();
    String json(Function<bool (const HeapSnapshotNode&)> allowNodeCallback);

    bool hasOverflowed() const { return m_hasOverflowed; }

private:
    static NodeIdentifier nextAvailableObjectIdentifier;
    static NodeIdentifier getNextObjectIdentifier();

    // Finalized snapshots are not modified during building. So searching them
    // for an existing node can be done concurrently without a lock.
    bool previousSnapshotHasNodeForCell(JSCell*, NodeIdentifier&);
    
    String descriptionForCell(JSCell*) const;
    
    struct RootData {
        ASCIILiteral reachabilityFromOpaqueRootReasons;
        RootMarkReason markReason { RootMarkReason::None };
    };
    
    HeapProfiler& m_profiler;
    OverflowPolicy m_overflowPolicy;
    bool m_hasOverflowed { false };

    // SlotVisitors run in parallel.
    Lock m_buildingNodeMutex;
    std::unique_ptr<HeapSnapshot> m_snapshot;
    Lock m_buildingEdgeMutex;
    Vector<HeapSnapshotEdge> m_edges;
    UncheckedKeyHashMap<JSCell*, RootData> m_rootData;
    UncheckedKeyHashMap<JSCell*, void*> m_wrappedObjectPointers;
    UncheckedKeyHashMap<JSCell*, String> m_cellLabels;
    UncheckedKeyHashSet<JSCell*> m_appendedCells;
    SnapshotType m_snapshotType;
};

} // namespace JSC
