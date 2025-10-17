/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#include "DFGConstantHoistingPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGInsertionSet.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

namespace {

class ConstantHoistingPhase : public Phase {
public:
    ConstantHoistingPhase(Graph& graph)
        : Phase(graph, "constant hoisting"_s)
    {
    }
    
    bool run()
    {
        DFG_ASSERT(m_graph, nullptr, m_graph.m_form == SSA);
        
        m_graph.clearReplacements();
        
        UncheckedKeyHashMap<FrozenValue*, Node*> jsValues;
        UncheckedKeyHashMap<FrozenValue*, Node*> doubleValues;
        UncheckedKeyHashMap<FrozenValue*, Node*> int52Values;
        
        auto valuesFor = [&] (NodeType op) -> UncheckedKeyHashMap<FrozenValue*, Node*>& {
            // Use a roundabout approach because clang thinks that this closure returning a
            // reference to a stack-allocated value in outer scope is a bug. It's not.
            UncheckedKeyHashMap<FrozenValue*, Node*>* result;
            
            switch (op) {
            case JSConstant:
                result = &jsValues;
                break;
            case DoubleConstant:
                result = &doubleValues;
                break;
            case Int52Constant:
                result = &int52Values;
                break;
            default:
                DFG_CRASH(m_graph, nullptr, "Invalid node type in valuesFor()");
                result = nullptr;
                break;
            }
            
            return *result;
        };
        
        Vector<Node*> toFree;
        
        for (BasicBlock* block : m_graph.blocksInNaturalOrder()) {
            unsigned sourceIndex = 0;
            unsigned targetIndex = 0;
            while (sourceIndex < block->size()) {
                Node* node = block->at(sourceIndex++);
                switch (node->op()) {
                case JSConstant:
                case DoubleConstant:
                case Int52Constant: {
                    UncheckedKeyHashMap<FrozenValue*, Node*>& values = valuesFor(node->op());
                    auto result = values.add(node->constant(), node);
                    if (result.isNewEntry)
                        node->origin = m_graph.block(0)->at(0)->origin;
                    else {
                        node->setReplacement(result.iterator->value);
                        toFree.append(node);
                    }
                    break;
                }
                default:
                    block->at(targetIndex++) = node;
                    break;
                }
            }
            block->resize(targetIndex);
        }
        
        // Insert the constants into the root block.
        InsertionSet insertionSet(m_graph);
        auto insertConstants = [&] (const UncheckedKeyHashMap<FrozenValue*, Node*>& values) {
            for (auto& entry : values)
                insertionSet.insert(0, entry.value);
        };
        insertConstants(jsValues);
        insertConstants(doubleValues);
        insertConstants(int52Values);
        insertionSet.execute(m_graph.block(0));
        
        // Perform all of the substitutions. We want all instances of the removed constants to
        // point at their replacements.
        for (BasicBlock* block : m_graph.blocksInNaturalOrder()) {
            for (Node* node : *block)
                m_graph.performSubstitution(node);
        }
        
        // And finally free the constants that we removed.
        m_graph.invalidateNodeLiveness();
        for (Node* node : toFree)
            m_graph.deleteNode(node);
        
        return true;
    }
};

} // anonymous namespace
    
bool performConstantHoisting(Graph& graph)
{
    return runPhase<ConstantHoistingPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

