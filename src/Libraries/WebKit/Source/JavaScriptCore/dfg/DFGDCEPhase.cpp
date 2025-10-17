/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#include "DFGDCEPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGInsertionSet.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class DCEPhase : public Phase {
public:
    DCEPhase(Graph& graph)
        : Phase(graph, "dead code elimination"_s)
        , m_insertionSet(graph)
    {
    }
    
    bool run()
    {
        ASSERT(m_graph.m_form == ThreadedCPS || m_graph.m_form == SSA);
        
        m_graph.computeRefCounts();
        
        for (BasicBlock* block : m_graph.blocksInPreOrder())
            fixupBlock(block);
        
        for (auto& argumentsVector : m_graph.m_rootToArguments.values())
            cleanVariables(argumentsVector);

        // Just do a basic Phantom/Check clean-up.
        for (BlockIndex blockIndex = m_graph.numBlocks(); blockIndex--;) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;
            unsigned sourceIndex = 0;
            unsigned targetIndex = 0;
            while (sourceIndex < block->size()) {
                Node* node = block->at(sourceIndex++);
                switch (node->op()) {
                case Check:
                case Phantom:
                    if (node->children.isEmpty())
                        continue;
                    break;
                case CheckVarargs: {
                    bool isEmpty = true;
                    m_graph.doToChildren(node, [&] (Edge edge) {
                        isEmpty &= !edge;
                    });
                    if (isEmpty)
                        continue;
                    break;
                }
                default:
                    break;
                }
                block->at(targetIndex++) = node;
            }
            block->resize(targetIndex);
        }
        
        m_graph.m_refCountState = ExactRefCount;
        
        return true;
    }

private:
    void fixupBlock(BasicBlock* block)
    {
        if (!block)
            return;

        if (m_graph.m_form == ThreadedCPS) {
            for (unsigned phiIndex = 0; phiIndex < block->phis.size(); ++phiIndex) {
                Node* phi = block->phis[phiIndex];
                if (!phi->shouldGenerate()) {
                    m_graph.deleteNode(phi);
                    block->phis[phiIndex--] = block->phis.last();
                    block->phis.removeLast();
                }
            }
            
            cleanVariables(block->variablesAtHead);
            cleanVariables(block->variablesAtTail);
        }

        // This has to be a forward loop because we are using the insertion set.
        for (unsigned indexInBlock = 0; indexInBlock < block->size(); ++indexInBlock) {
            Node* node = block->at(indexInBlock);
            if (node->shouldGenerate())
                continue;
                
            if (node->flags() & NodeHasVarArgs) {
                for (unsigned childIdx = node->firstChild(); childIdx < node->firstChild() + node->numChildren(); childIdx++) {
                    Edge edge = m_graph.m_varArgChildren[childIdx];
                    
                    if (!edge || edge.willNotHaveCheck())
                        continue;
                    
                    m_insertionSet.insertNode(indexInBlock, SpecNone, Check, node->origin, edge);
                }
                
                node->setOpAndDefaultFlags(Check);
                node->children.reset();
                node->setRefCount(1);
                continue;
            }
            
            node->remove(m_graph);
            node->setRefCount(1);
        }

        m_insertionSet.execute(block);
    }
    
    template<typename VariablesVectorType>
    void cleanVariables(VariablesVectorType& variables)
    {
        for (unsigned i = variables.size(); i--;) {
            Node* node = variables[i];
            if (!node)
                continue;
            if (node->op() != Check && node->shouldGenerate())
                continue;
            variables[i] = nullptr;
        }
    }
    
    InsertionSet m_insertionSet;
};

bool performDCE(Graph& graph)
{
    return runPhase<DCEPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

