/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#include "DFGOSREntrypointCreationPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGBasicBlockInlines.h"
#include "DFGBlockInsertionSet.h"
#include "DFGGraph.h"
#include "DFGLoopPreHeaderCreationPhase.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class OSREntrypointCreationPhase : public Phase {
public:
    OSREntrypointCreationPhase(Graph& graph)
        : Phase(graph, "OSR entrypoint creation"_s)
    {
    }
    
    bool run()
    {
        RELEASE_ASSERT(m_graph.m_plan.mode() == JITCompilationMode::FTLForOSREntry);
        RELEASE_ASSERT(m_graph.m_form == ThreadedCPS);

        BytecodeIndex bytecodeIndex = m_graph.m_plan.osrEntryBytecodeIndex();
        RELEASE_ASSERT(bytecodeIndex);
        RELEASE_ASSERT(bytecodeIndex.offset());

        // Needed by createPreHeader().
        m_graph.ensureCPSDominators();
        
        CodeBlock* baseline = m_graph.m_profiledBlock;
        
        BasicBlock* target = nullptr;
        for (unsigned blockIndex = m_graph.numBlocks(); blockIndex--;) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;
            unsigned nodeIndex = 0;
            Node* firstNode = block->at(0);
            while (firstNode->isSemanticallySkippable())
                firstNode = block->at(++nodeIndex);
            if (firstNode->op() == LoopHint
                && firstNode->origin.semantic == CodeOrigin(bytecodeIndex)) {
                target = block;
                break;
            }
        }

        if (!target) {
            // This is a terrible outcome. It shouldn't often happen but it might
            // happen and so we should defend against it. If it happens, then this
            // compilation is a failure.
            return false;
        }
        
        BlockInsertionSet insertionSet(m_graph);
        
        // We say that the execution count of the entry block is 1, because we know for sure
        // that this must be the case. Under our definition of executionCount, "1" means "once
        // per invocation". We could have said NaN here, since that would ask any clients of
        // executionCount to use best judgement - but that seems unnecessary since we know for
        // sure what the executionCount should be in this case.
        BasicBlock* newRoot = insertionSet.insert(0, 1);

        // We'd really like to use an unset origin, but ThreadedCPS won't allow that.
        NodeOrigin origin = NodeOrigin(CodeOrigin(BytecodeIndex(0)), CodeOrigin(BytecodeIndex(0)), false);
        
        Vector<Node*> locals(baseline->numCalleeLocals());
        for (unsigned local = 0; local < baseline->numCalleeLocals(); ++local) {
            Node* previousHead = target->variablesAtHead.local(local);
            if (!previousHead)
                continue;
            VariableAccessData* variable = previousHead->variableAccessData();
            locals[local] = newRoot->appendNode(
                m_graph, variable->prediction(), ExtractOSREntryLocal, origin,
                OpInfo(variable->operand().virtualRegister()));
            
            newRoot->appendNode(
                m_graph, SpecNone, MovHint, origin, OpInfo(variable->operand().virtualRegister()),
                Edge(locals[local]));
        }

        // Now use the origin of the target, since it's not OK to exit, and we will probably hoist
        // type checks to here.
        origin = target->at(0)->origin;
        
        ArgumentsVector newArguments = m_graph.m_rootToArguments.find(m_graph.block(0))->value;
        for (unsigned argument = 0; argument < baseline->numParameters(); ++argument) {
            Node* oldNode = target->variablesAtHead.argument(argument);
            if (!oldNode) {
                // Just for sanity, always have a SetArgumentDefinitely even if it's not needed.
                oldNode = newArguments[argument];
            }
            Node* node = newRoot->appendNode(
                m_graph, SpecNone, SetArgumentDefinitely, origin,
                OpInfo(oldNode->variableAccessData()));
            newArguments[argument] = node;
        }

        for (unsigned local = 0; local < baseline->numCalleeLocals(); ++local) {
            Node* previousHead = target->variablesAtHead.local(local);
            if (!previousHead)
                continue;
            VariableAccessData* variable = previousHead->variableAccessData();
            Node* node = locals[local];
            newRoot->appendNode(
                m_graph, SpecNone, SetLocal, origin, OpInfo(variable), Edge(node));
        }
        
        newRoot->appendNode(
            m_graph, SpecNone, Jump, origin,
            OpInfo(createPreHeader(m_graph, insertionSet, target)));
        
        insertionSet.execute();

        RELEASE_ASSERT(m_graph.m_roots.size() == 1);
        m_graph.m_roots[0] = newRoot;
        m_graph.m_rootToArguments.clear();
        m_graph.m_rootToArguments.add(newRoot, newArguments);

        m_graph.invalidateCFG();
        m_graph.resetReachability();
        m_graph.killUnreachableBlocks();

        return true;
    }
};

bool performOSREntrypointCreation(Graph& graph)
{
    return runPhase<OSREntrypointCreationPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)


