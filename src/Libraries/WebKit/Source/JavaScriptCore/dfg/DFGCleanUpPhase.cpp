/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
#include "DFGCleanUpPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class CleanUpPhase : public Phase {
public:
    CleanUpPhase(Graph& graph)
        : Phase(graph, "clean up"_s)
    {
    }
    
    bool run()
    {
        bool changed = false;
        
        for (BasicBlock* block : m_graph.blocksInNaturalOrder()) {
            unsigned sourceIndex = 0;
            unsigned targetIndex = 0;
            while (sourceIndex < block->size()) {
                Node* node = block->at(sourceIndex++);
                bool kill = false;
                
                if (node->op() == Check)
                    node->children = node->children.justChecks();
                
                switch (node->op()) {
                case Phantom:
                case Check:
                    if (node->children.isEmpty())
                        kill = true;
                    break;
                case CheckVarargs:
                    kill = true;
                    m_graph.doToChildren(node, [&] (Edge edge) {
                        kill &= !edge;
                    });
                    break;
                default:
                    break;
                }
                
                if (kill)
                    m_graph.deleteNode(node);
                else
                    block->at(targetIndex++) = node;
            }
            block->resize(targetIndex);
        }
        
        return changed;
    }
};
    
bool performCleanUp(Graph& graph)
{
    return runPhase<CleanUpPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

