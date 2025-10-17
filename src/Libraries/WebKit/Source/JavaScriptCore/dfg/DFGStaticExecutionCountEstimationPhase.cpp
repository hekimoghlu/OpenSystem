/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#include "DFGStaticExecutionCountEstimationPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGNaturalLoops.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class StaticExecutionCountEstimationPhase : public Phase {
public:
    StaticExecutionCountEstimationPhase(Graph& graph)
        : Phase(graph, "static execution count estimation"_s)
    {
    }
    
    bool run()
    {
        m_graph.ensureCPSNaturalLoops();
        
        // Estimate basic block execution counts based on loop depth.
        for (BlockIndex blockIndex = m_graph.numBlocks(); blockIndex--;) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;

            block->executionCount = pow(10, m_graph.m_cpsNaturalLoops->loopDepth(block));
        }
        
        // Estimate branch weights based on execution counts. This isn't quite correct. It'll
        // assume that each block's conditional successor only has that block as its
        // predecessor.
        for (BlockIndex blockIndex = m_graph.numBlocks(); blockIndex--;) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;
            
            Node* terminal = block->terminal();
            switch (terminal->op()) {
            case Branch: {
                BranchData* data = terminal->branchData();
                applyCounts(data->taken);
                applyCounts(data->notTaken);
                break;
            }
                
            case Switch: {
                SwitchData* data = terminal->switchData();
                for (unsigned i = data->cases.size(); i--;)
                    applyCounts(data->cases[i].target);
                applyCounts(data->fallThrough);
                break;
            }
            
            case EntrySwitch: {
                DFG_CRASH(m_graph, terminal, "Unexpected EntrySwitch in CPS form.");
                break;
            }

            default:
                break;
            }
        }
        
        return true;
    }

private:
    void applyCounts(BranchTarget& target)
    {
        target.count = target.block->executionCount;
    }
};

bool performStaticExecutionCountEstimation(Graph& graph)
{
    return runPhase<StaticExecutionCountEstimationPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)


