/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
#include "DFGAvailabilityMap.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "JSCJSValueInlines.h"
#include "OperandsInlines.h"
#include <wtf/ListDump.h>

namespace JSC { namespace DFG {

void AvailabilityMap::pruneHeap()
{
    if (m_heap.isEmpty())
        return;
    
    NodeSet possibleNodes;
    
    for (unsigned i = m_locals.size(); i--;) {
        if (m_locals[i].hasNode())
            possibleNodes.addVoid(m_locals[i].node());
    }

    closeOverNodes(
        [&] (Node* node) -> bool {
            return possibleNodes.contains(node);
        },
        [&] (Node* node) -> bool {
            return possibleNodes.add(node).isNewEntry;
        });
    
    UncheckedKeyHashMap<PromotedHeapLocation, Availability> newHeap;
    for (auto pair : m_heap) {
        if (possibleNodes.contains(pair.key.base()))
            newHeap.add(pair.key, pair.value);
    }
    m_heap = WTFMove(newHeap);
}

void AvailabilityMap::pruneByLiveness(Graph& graph, CodeOrigin where)
{
    Operands<Availability> localsCopy(OperandsLike, m_locals, Availability::unavailable());
    graph.forAllLiveInBytecode(
        where,
        [&] (Operand reg) {
            localsCopy.operand(reg) = m_locals.operand(reg);
        });
    m_locals = WTFMove(localsCopy);
    pruneHeap();
}

void AvailabilityMap::clear()
{
    m_locals.fill(Availability());
    m_heap.clear();
}

void AvailabilityMap::dump(PrintStream& out) const
{
    out.print("{locals = ", m_locals, "; heap = ", mapDump(m_heap), "}");
}

void AvailabilityMap::merge(const AvailabilityMap& other)
{
    for (unsigned i = other.m_locals.size(); i--;)
        m_locals[i] = other.m_locals[i].merge(m_locals[i]);
    
    for (auto pair : other.m_heap) {
        auto result = m_heap.add(pair.key, Availability());
        result.iterator->value = pair.value.merge(result.iterator->value);
    }
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

