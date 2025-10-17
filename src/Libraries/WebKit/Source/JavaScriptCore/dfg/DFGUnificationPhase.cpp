/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
#include "DFGUnificationPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class UnificationPhase : public Phase {
public:
    UnificationPhase(Graph& graph)
        : Phase(graph, "unification"_s)
    {
    }
    
    bool run()
    {
        ASSERT(m_graph.m_form == ThreadedCPS);
        ASSERT(m_graph.m_unificationState == LocallyUnified);
        
        // Ensure that all Phi functions are unified.
        for (BlockIndex blockIndex = m_graph.numBlocks(); blockIndex--;) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;
            ASSERT(block->isReachable);
            
            for (unsigned phiIndex = block->phis.size(); phiIndex--;) {
                Node* phi = block->phis[phiIndex];
                for (unsigned childIdx = 0; childIdx < AdjacencyList::Size; ++childIdx) {
                    if (!phi->children.child(childIdx))
                        break;

                    // FIXME: Consider reversing the order of this unification, since the other
                    // order will reveal more bugs. https://bugs.webkit.org/show_bug.cgi?id=154368
                    phi->variableAccessData()->unify(phi->children.child(childIdx)->variableAccessData());
                }
            }
        }
        
        // Ensure that all predictions are fixed up based on the unification.
        for (unsigned i = 0; i < m_graph.m_variableAccessData.size(); ++i) {
            VariableAccessData* data = &m_graph.m_variableAccessData[i];
            data->find()->predict(data->nonUnifiedPrediction());
            data->find()->mergeStructureCheckHoistingFailed(data->structureCheckHoistingFailed());
            data->find()->mergeCheckArrayHoistingFailed(data->checkArrayHoistingFailed());
            data->find()->mergeShouldNeverUnbox(data->shouldNeverUnbox());
            data->find()->mergeIsLoadedFrom(data->isLoadedFrom());
            data->find()->mergeIsProfitableToUnbox(data->isProfitableToUnbox());
            data->find()->mergeFlags(data->flags());
        }
        
        m_graph.m_unificationState = GloballyUnified;
        return true;
    }
};

bool performUnification(Graph& graph)
{
    return runPhase<UnificationPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

