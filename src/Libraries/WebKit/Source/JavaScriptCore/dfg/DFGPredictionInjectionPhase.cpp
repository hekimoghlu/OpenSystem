/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#include "DFGPredictionInjectionPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class PredictionInjectionPhase : public Phase {
public:
    PredictionInjectionPhase(Graph& graph)
        : Phase(graph, "prediction injection"_s)
    {
    }
    
    bool run()
    {
        ASSERT(m_graph.m_form == ThreadedCPS);
        ASSERT(m_graph.m_unificationState == GloballyUnified);
        
        ASSERT(codeBlock()->numParameters() >= 1);
        {
            ConcurrentJSLocker locker(profiledBlock()->valueProfileLock());
            
            // We only do this for the arguments at the first block. The arguments from
            // other entrypoints have already been populated with their predictions.
            auto& arguments = m_graph.m_rootToArguments.find(m_graph.block(0))->value;

            for (size_t arg = 0; arg < static_cast<size_t>(codeBlock()->numParameters()); ++arg) {
                ArgumentValueProfile& profile = profiledBlock()->valueProfileForArgument(arg);
                arguments[arg]->variableAccessData()->predict(profile.computeUpdatedPrediction(locker));
            }
        }
        
        for (BlockIndex blockIndex = 0; blockIndex < m_graph.numBlocks(); ++blockIndex) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;
            if (!block->isOSRTarget)
                continue;
            if (block->bytecodeBegin != m_graph.m_plan.osrEntryBytecodeIndex())
                continue;
            const Operands<std::optional<JSValue>>& mustHandleValues = m_graph.m_plan.mustHandleValues();
            for (size_t i = 0; i < mustHandleValues.size(); ++i) {
                Operand operand = mustHandleValues.operandForIndex(i);
                std::optional<JSValue> value = mustHandleValues[i];
                if (!value)
                    continue;
                Node* node = block->variablesAtHead.operand(operand);
                if (!node)
                    continue;
                ASSERT(node->accessesStack(m_graph));
                node->variableAccessData()->predict(speculationFromValue(value.value()));
            }
        }
        
        return true;
    }
};

bool performPredictionInjection(Graph& graph)
{
    return runPhase<PredictionInjectionPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

