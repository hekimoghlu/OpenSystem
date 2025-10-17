/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#include "DFGVirtualRegisterAllocationPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGScoreBoard.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class VirtualRegisterAllocationPhase : public Phase {
public:
    VirtualRegisterAllocationPhase(Graph& graph)
        : Phase(graph, "virtual register allocation"_s)
    {
    }
    
    bool run()
    {
        DFG_ASSERT(m_graph, nullptr, m_graph.m_form == ThreadedCPS);
        
        ScoreBoard scoreBoard(m_graph.m_nextMachineLocal);
        scoreBoard.assertClear();
        for (size_t blockIndex = 0; blockIndex < m_graph.numBlocks(); ++blockIndex) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;
            if (!block->isReachable)
                continue;
            if (ASSERT_ENABLED) {
                // Force usage of highest-numbered virtual registers.
                scoreBoard.sortFree();
            }
            for (auto* node : *block) {
                if (!node->shouldGenerate())
                    continue;
                
                switch (node->op()) {
                case Phi:
                case Flush:
                case PhantomLocal:
                    continue;
                case GetLocal:
                    ASSERT(!node->child1()->hasResult());
                    break;

                case ExtractFromTuple:
                    ASSERT(m_graph.m_tupleData.at(node->tupleIndex()).refCount == 1);
                    node->setVirtualRegister(m_graph.m_tupleData.at(node->tupleIndex()).virtualRegister);
                    ASSERT(!node->mustGenerate());
                    continue;

                default:
                    break;
                }
                
                // First, call use on all of the current node's children, then
                // allocate a VirtualRegister for this node. We do so in this
                // order so that if a child is on its last use, and a
                // VirtualRegister is freed, then it may be reused for node.
                if (node->flags() & NodeHasVarArgs) {
                    for (unsigned childIdx = node->firstChild(); childIdx < node->firstChild() + node->numChildren(); childIdx++)
                        scoreBoard.useIfHasResult(m_graph.m_varArgChildren[childIdx]);
                } else {
                    scoreBoard.useIfHasResult(node->child1());
                    scoreBoard.useIfHasResult(node->child2());
                    scoreBoard.useIfHasResult(node->child3());
                }

                if (node->isTuple()) {
                    ASSERT(node->refCount() <= node->tupleSize());
                    for (unsigned i = 0; i < node->tupleSize(); ++i) {
                        auto& tupleData = m_graph.m_tupleData.at(node->tupleOffset() + i);
                        if (tupleData.refCount)
                            tupleData.virtualRegister = scoreBoard.allocate();
                    }
                    ASSERT(!node->hasResult());
                    continue;
                }

                if (!node->hasResult())
                    continue;

                VirtualRegister virtualRegister = scoreBoard.allocate();
                node->setVirtualRegister(virtualRegister);
                // 'mustGenerate' nodes have their useCount artificially elevated,
                // call use now to account for this.
                if (node->mustGenerate())
                    scoreBoard.use(node);
            }
            scoreBoard.assertClear();
        }
        
        // Record the number of virtual registers we're using. This is used by calls
        // to figure out where to put the parameters.
        m_graph.m_nextMachineLocal = scoreBoard.highWatermark();

        return true;
    }
};

bool performVirtualRegisterAllocation(Graph& graph)
{
    return runPhase<VirtualRegisterAllocationPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
