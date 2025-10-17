/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#include "DFGFlowIndexing.h"

#if ENABLE(DFG_JIT)

#include "JSCJSValueInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace DFG {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FlowIndexing);

FlowIndexing::FlowIndexing(Graph& graph)
    : m_graph(graph)
    , m_numIndices(0)
{
    recompute();
}

FlowIndexing::~FlowIndexing() = default;

void FlowIndexing::recompute()
{
    unsigned numNodeIndices = m_graph.maxNodeCount();
    
    m_nodeIndexToShadowIndex.resize(numNodeIndices);
    m_nodeIndexToShadowIndex.fill(UINT_MAX);
    
    m_shadowIndexToNodeIndex.shrink(0);

    m_numIndices = numNodeIndices;

    for (BasicBlock* block : m_graph.blocksInNaturalOrder()) {
        for (Node* node : *block) {
            if (node->op() != Phi)
                continue;
            
            unsigned nodeIndex = node->index();
            unsigned shadowIndex = m_numIndices++;
            m_nodeIndexToShadowIndex[nodeIndex] = shadowIndex;
            m_shadowIndexToNodeIndex.append(nodeIndex);
            DFG_ASSERT(m_graph, nullptr, m_shadowIndexToNodeIndex.size() + numNodeIndices == m_numIndices);
            DFG_ASSERT(m_graph, nullptr, m_shadowIndexToNodeIndex[shadowIndex - numNodeIndices] == nodeIndex);
        }
    }
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

