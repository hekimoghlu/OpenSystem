/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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
#include "DFGAtTailAbstractState.h"
#include "DFGBlockMapInlines.h"

#if ENABLE(DFG_JIT)

#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

AtTailAbstractState::AtTailAbstractState(Graph& graph)
    : m_graph(graph)
    , m_valuesAtTailMap(m_graph)
    , m_tupleAbstractValues(m_graph)
{
    for (BasicBlock* block : graph.blocksInNaturalOrder()) {
        auto& valuesAtTail = m_valuesAtTailMap.at(block);
        valuesAtTail.clear();
        for (auto& valueAtTailPair : block->ssa->valuesAtTail)
            valuesAtTail.add(valueAtTailPair.node, valueAtTailPair.value);
        m_tupleAbstractValues.at(block).grow(m_graph.m_tupleData.size());
    }
}

AtTailAbstractState::~AtTailAbstractState() = default;

void AtTailAbstractState::createValueForNode(NodeFlowProjection node)
{
    m_valuesAtTailMap.at(m_block).add(node, AbstractValue());
}

AbstractValue& AtTailAbstractState::forNode(NodeFlowProjection node)
{
    ASSERT(!node->isTuple());
    auto& valuesAtTail = m_valuesAtTailMap.at(m_block);
    UncheckedKeyHashMap<NodeFlowProjection, AbstractValue>::iterator iter = valuesAtTail.find(node);
    DFG_ASSERT(m_graph, node.node(), iter != valuesAtTail.end());
    return iter->value;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

