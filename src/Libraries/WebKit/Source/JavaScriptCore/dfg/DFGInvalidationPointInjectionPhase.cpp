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
#include "DFGInvalidationPointInjectionPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGBlockSetInlines.h"
#include "DFGClobberize.h"
#include "DFGGraph.h"
#include "DFGInsertionSet.h"
#include "DFGPhase.h"

namespace JSC { namespace DFG {

class InvalidationPointInjectionPhase : public Phase {
    static constexpr bool verbose = false;
    
public:
    InvalidationPointInjectionPhase(Graph& graph)
        : Phase(graph, "invalidation point injection"_s)
        , m_insertionSet(graph)
    {
    }
    
    bool run()
    {
        ASSERT(m_graph.m_form != SSA);
        
        BlockSet blocksThatNeedInvalidationPoints;
        
        for (BlockIndex blockIndex = m_graph.numBlocks(); blockIndex--;) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;
            
            m_originThatHadFire = CodeOrigin();
            
            for (unsigned nodeIndex = 0; nodeIndex < block->size(); ++nodeIndex)
                handle(nodeIndex, block->at(nodeIndex));
            
            // Note: this assumes that control flow occurs at bytecode instruction boundaries.
            if (m_originThatHadFire.isSet()) {
                for (unsigned i = block->numSuccessors(); i--;)
                    blocksThatNeedInvalidationPoints.add(block->successor(i));
            }
            
            m_insertionSet.execute(block);
        }

        for (BasicBlock* block : blocksThatNeedInvalidationPoints.iterable(m_graph)) {
            insertInvalidationCheck(0, block->at(0));
            m_insertionSet.execute(block);
        }
        
        return true;
    }

private:
    void handle(unsigned nodeIndex, Node* node)
    {
        if (m_originThatHadFire.isSet() && m_originThatHadFire != node->origin.forExit) {
            insertInvalidationCheck(nodeIndex, node);
            m_originThatHadFire = CodeOrigin();
        }
        
        if (writesOverlap(m_graph, node, Watchpoint_fire))
            m_originThatHadFire = node->origin.forExit;
    }
    
    void insertInvalidationCheck(unsigned nodeIndex, Node* node)
    {
        m_insertionSet.insertNode(nodeIndex, SpecNone, InvalidationPoint, node->origin);
    }
    
    CodeOrigin m_originThatHadFire;
    InsertionSet m_insertionSet;
};

bool performInvalidationPointInjection(Graph& graph)
{
    return runPhase<InvalidationPointInjectionPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

