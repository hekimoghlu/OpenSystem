/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#include "DFGClobberSet.h"

#if ENABLE(DFG_JIT)

#include "ButterflyInlines.h"
#include "DFGClobberize.h"
#include <wtf/ListDump.h>

namespace JSC { namespace DFG {

ClobberSet::ClobberSet() = default;
ClobberSet::~ClobberSet() = default;

void ClobberSet::add(AbstractHeap heap)
{
    UncheckedKeyHashMap<AbstractHeap, bool>::AddResult result = m_clobbers.add(heap, true);
    if (!result.isNewEntry) {
        if (result.iterator->value)
            return;
        result.iterator->value = true;
    }
    while (heap.kind() != World) {
        heap = heap.supertype();
        if (!m_clobbers.add(heap, false).isNewEntry)
            return;
    }
}

void ClobberSet::addAll(const ClobberSet& other)
{
    // If the other set has a direct heap, we make sure we have it and we set its
    // value to be true.
    //
    // If the other heap has a super heap, we make sure it's present but don't
    // modify its value - so we had it directly already then this doesn't change.
    
    if (this == &other)
        return;
    
    UncheckedKeyHashMap<AbstractHeap, bool>::const_iterator iter = other.m_clobbers.begin();
    UncheckedKeyHashMap<AbstractHeap, bool>::const_iterator end = other.m_clobbers.end();
    for (; iter != end; ++iter)
        m_clobbers.add(iter->key, iter->value).iterator->value |= iter->value;
}

bool ClobberSet::contains(AbstractHeap heap) const
{
    UncheckedKeyHashMap<AbstractHeap, bool>::const_iterator iter = m_clobbers.find(heap);
    if (iter == m_clobbers.end())
        return false;
    return iter->value;
}

bool ClobberSet::overlaps(AbstractHeap heap) const
{
    if (m_clobbers.find(heap) != m_clobbers.end())
        return true;
    if (heap.kind() == DOMState && !heap.payload().isTop()) {
        // DOMState heap has its own hierarchy. For direct heap clobbers that payload is not Top,
        // we should query whether the clobber overlaps with the given heap.
        DOMJIT::HeapRange range = DOMJIT::HeapRange::fromRaw(heap.payload().value32());
        for (auto pair : m_clobbers) {
            bool direct = pair.value;
            if (!direct)
                continue;
            AbstractHeap clobber = pair.key;
            if (clobber.kind() != DOMState)
                continue;
            if (clobber.payload().isTop())
                return true;
            if (DOMJIT::HeapRange::fromRaw(clobber.payload().value32()).overlaps(range))
                return true;
        }
    }
    while (heap.kind() != World) {
        heap = heap.supertype();
        if (contains(heap))
            return true;
    }
    return false;
}

void ClobberSet::clear()
{
    m_clobbers.clear();
}

UncheckedKeyHashSet<AbstractHeap> ClobberSet::direct() const
{
    return setOf(true);
}

UncheckedKeyHashSet<AbstractHeap> ClobberSet::super() const
{
    return setOf(false);
}

void ClobberSet::dump(PrintStream& out) const
{
    out.print("(Direct:[", sortedListDump(direct()), "], Super:[", sortedListDump(super()), "])");
}

UncheckedKeyHashSet<AbstractHeap> ClobberSet::setOf(bool direct) const
{
    UncheckedKeyHashSet<AbstractHeap> result;
    for (auto& clobber : m_clobbers) {
        if (clobber.value == direct)
            result.add(clobber.key);
    }
    return result;
}

void addReads(Graph& graph, Node* node, ClobberSet& readSet)
{
    ClobberSetAdd addRead(readSet);
    NoOpClobberize noOp;
    clobberize(graph, node, addRead, noOp, noOp);
}

void addWrites(Graph& graph, Node* node, ClobberSet& writeSet)
{
    NoOpClobberize noOp;
    ClobberSetAdd addWrite(writeSet);
    clobberize(graph, node, noOp, addWrite, noOp);
}

void addReadsAndWrites(Graph& graph, Node* node, ClobberSet& readSet, ClobberSet& writeSet)
{
    ClobberSetAdd addRead(readSet);
    ClobberSetAdd addWrite(writeSet);
    NoOpClobberize noOp;
    clobberize(graph, node, addRead, addWrite, noOp);
}

ClobberSet writeSet(Graph& graph, Node* node)
{
    ClobberSet result;
    addWrites(graph, node, result);
    return result;
}

bool readsOverlap(Graph& graph, Node* node, ClobberSet& readSet)
{
    ClobberSetOverlaps addRead(readSet);
    NoOpClobberize noOp;
    clobberize(graph, node, addRead, noOp, noOp);
    return addRead.result();
}

bool writesOverlap(Graph& graph, Node* node, ClobberSet& writeSet)
{
    NoOpClobberize noOp;
    ClobberSetOverlaps addWrite(writeSet);
    clobberize(graph, node, noOp, addWrite, noOp);
    return addWrite.result();
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

