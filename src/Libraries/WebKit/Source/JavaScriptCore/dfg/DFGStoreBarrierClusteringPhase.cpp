/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#include "DFGStoreBarrierClusteringPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGDoesGC.h"
#include "DFGGraph.h"
#include "DFGInsertionSet.h"
#include "DFGMayExit.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"
#include <wtf/FastBitVector.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace DFG {

namespace {

constexpr bool verbose = false;

class StoreBarrierClusteringPhase : public Phase {
public:
    StoreBarrierClusteringPhase(Graph& graph)
        : Phase(graph, "store barrier clustering"_s)
        , m_insertionSet(graph)
    {
    }
    
    bool run()
    {
        size_t maxSize = 0;
        for (BasicBlock* block : m_graph.blocksInNaturalOrder())
            maxSize = std::max(maxSize, block->size());
        m_barrierPoints.resize(maxSize);
        
        for (BasicBlock* block : m_graph.blocksInNaturalOrder()) {
            size_t blockSize = block->size();
            doBlock(block);
            m_barrierPoints.clearRange(0, blockSize);
        }
        
        return true;
    }

private:
    // This summarizes everything we need to remember about a barrier.
    struct ChildAndOrigin {
        ChildAndOrigin() { }
        
        ChildAndOrigin(Node* child, CodeOrigin semanticOrigin)
            : child(child)
            , semanticOrigin(semanticOrigin)
        {
        }
        
        Node* child { nullptr };
        CodeOrigin semanticOrigin;
    };
    
    void doBlock(BasicBlock* block)
    {
        ASSERT(m_barrierPoints.isEmpty());
        
        // First identify the places where we would want to place all of the barriers based on a
        // backwards analysis. We use the futureGC flag to tell us if we had seen a GC. Since this
        // is a backwards analysis, when we get to a node, futureGC tells us if a GC will happen
        // in the future after that node.
        bool futureGC = true;
        for (unsigned nodeIndex = block->size(); nodeIndex--;) {
            Node* node = block->at(nodeIndex);
            
            // This is a backwards analysis, so exits require conservatism. If we exit, then there
            // probably will be a GC in the future! If we needed to then we could lift that
            // requirement by either (1) having a StoreBarrierHint that tells OSR exit to barrier that
            // value or (2) automatically barriering any DFG-live Node on OSR exit. Either way, it
            // would be weird because it would create a new root for OSR availability analysis. I
            // don't have evidence that it would be worth it.
            if (doesGC(m_graph, node) || mayExit(m_graph, node) != DoesNotExit) {
                if (verbose) {
                    dataLog("Possible GC point at ", node, "\n");
                    dataLog("    doesGC = ", doesGC(m_graph, node), "\n");
                    dataLog("    mayExit = ", mayExit(m_graph, node), "\n");
                }
                futureGC = true;
                continue;
            }
            
            if (node->isStoreBarrier() && futureGC) {
                m_barrierPoints[nodeIndex] = true;
                futureGC = false;
            }
        }
        
        // Now we run forward and collect the barriers. When we hit a barrier point, insert all of
        // them with a fence.
        for (unsigned nodeIndex = 0; nodeIndex < block->size(); ++nodeIndex) {
            Node* node = block->at(nodeIndex);
            if (!node->isStoreBarrier())
                continue;
            
            DFG_ASSERT(m_graph, node, !node->origin.wasHoisted);
            DFG_ASSERT(m_graph, node, node->child1().useKind() == KnownCellUse, node->op(), node->child1().useKind());
            
            NodeOrigin origin = node->origin;
            m_neededBarriers.append(ChildAndOrigin(node->child1().node(), origin.semantic));
            node->remove(m_graph);
            
            if (!m_barrierPoints[nodeIndex])
                continue;
            
            std::sort(
                m_neededBarriers.begin(), m_neededBarriers.end(),
                [&] (const ChildAndOrigin& a, const ChildAndOrigin& b) -> bool {
                    return a.child < b.child;
                });
            removeRepeatedElements(
                m_neededBarriers, 
                [&] (const ChildAndOrigin& a, const ChildAndOrigin& b) -> bool{
                    return a.child == b.child;
                });
            for (auto iter = m_neededBarriers.begin(); iter != m_neededBarriers.end(); ++iter) {
                Node* child = iter->child;
                CodeOrigin semanticOrigin = iter->semanticOrigin;
                
                NodeType type;
                if (iter == m_neededBarriers.begin())
                    type = FencedStoreBarrier;
                else
                    type = StoreBarrier;
                
                m_insertionSet.insertNode(
                    nodeIndex, SpecNone, type, origin.withSemantic(semanticOrigin),
                    Edge(child, KnownCellUse));
            }
            m_neededBarriers.shrink(0);
        }
        
        m_insertionSet.execute(block);
    }
    
    InsertionSet m_insertionSet;
    FastBitVector m_barrierPoints;
    Vector<ChildAndOrigin> m_neededBarriers;
};

} // anonymous namespace

bool performStoreBarrierClustering(Graph& graph)
{
    return runPhase<StoreBarrierClusteringPhase>(graph);
}

} } // namespace JSC::DFG

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(DFG_JIT)
