/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#include "DFGSSALoweringPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGInsertionSet.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class SSALoweringPhase : public Phase {
    static constexpr bool verbose = false;
    
public:
    SSALoweringPhase(Graph& graph)
        : Phase(graph, "SSA lowering"_s)
        , m_insertionSet(graph)
    {
    }
    
    bool run()
    {
        RELEASE_ASSERT(m_graph.m_form == SSA);
        
        for (BlockIndex blockIndex = m_graph.numBlocks(); blockIndex--;) {
            m_block = m_graph.block(blockIndex);
            if (!m_block)
                continue;
            for (m_nodeIndex = 0; m_nodeIndex < m_block->size(); ++m_nodeIndex) {
                m_node = m_block->at(m_nodeIndex);
                handleNode();
            }
            m_insertionSet.execute(m_block);
        }

        return true;
    }

private:
    void handleNode()
    {
        switch (m_node->op()) {
        case AtomicsAdd:
        case AtomicsAnd:
        case AtomicsCompareExchange:
        case AtomicsExchange:
        case AtomicsLoad:
        case AtomicsOr:
        case AtomicsStore:
        case AtomicsSub:
        case AtomicsXor: {
            unsigned numExtraArgs = numExtraAtomicsArgs(m_node->op());
            lowerBoundsCheck(m_graph.child(m_node, 0), m_graph.child(m_node, 1), m_graph.child(m_node, 2 + numExtraArgs));
            break;
        }

        case HasIndexedProperty:
            lowerBoundsCheck(m_graph.child(m_node, 0), m_graph.child(m_node, 1), m_graph.child(m_node, 2));
            break;

        case EnumeratorGetByVal:
        case GetByVal: {
            lowerBoundsCheck(m_graph.varArgChild(m_node, 0), m_graph.varArgChild(m_node, 1), m_graph.varArgChild(m_node, 2));
            break;
        }
            
        case EnumeratorPutByVal:
            break;

        case PutByVal:
        case PutByValDirect: {
            Edge base = m_graph.varArgChild(m_node, 0);
            Edge index = m_graph.varArgChild(m_node, 1);
            Edge storage = m_graph.varArgChild(m_node, 3);
            if (lowerBoundsCheck(base, index, storage))
                break;
            
            if (m_node->arrayMode().isSomeTypedArrayView() && m_node->arrayMode().isOutOfBounds()) {
#if USE(LARGE_TYPED_ARRAYS)
                if (m_node->arrayMode().mayBeLargeTypedArray() || m_graph.hasExitSite(m_node->origin.semantic, Overflow)) {
                    Node* length = m_insertionSet.insertNode(
                        m_nodeIndex, SpecInt52Any, GetTypedArrayLengthAsInt52, m_node->origin,
                        OpInfo(m_node->arrayMode().asWord()), base, storage);
                    length->setResult(NodeResultInt52);
                    // GetTypedArrayLengthAsInt52 says write(MiscFields) to model concurrent updates. But this does not mean that
                    // we cannot exit after running GetTypedArrayLengthAsInt52 since exit state is still intact after that.
                    // To teach DFG / FTL about it, we insert ExitOK node here to make subsequent nodes valid for exits.
                    if (m_node->arrayMode().mayBeResizableOrGrowableSharedTypedArray())
                        m_insertionSet.insertNode(m_nodeIndex, SpecNone, ExitOK, m_node->origin.withExitOK(true));
                    m_graph.varArgChild(m_node, 4) = Edge(length, Int52RepUse);
                } else {
#endif
                    Node* length = m_insertionSet.insertNode(
                        m_nodeIndex, SpecInt32Only, GetArrayLength, m_node->origin,
                        OpInfo(m_node->arrayMode().asWord()), base, storage);
                    if (m_node->arrayMode().mayBeResizableOrGrowableSharedTypedArray())
                        m_insertionSet.insertNode(m_nodeIndex, SpecNone, ExitOK, m_node->origin.withExitOK(true));
                    m_graph.varArgChild(m_node, 4) = Edge(length, KnownInt32Use);
#if USE(LARGE_TYPED_ARRAYS)
                }
#endif
                break;
            }
            break;
        }
            
        default:
            break;
        }
    }
    
    bool lowerBoundsCheck(Edge base, Edge index, Edge storage)
    {
        if (!m_node->arrayMode().permitsBoundsCheckLowering())
            return false;
        
        if (!m_node->arrayMode().lengthNeedsStorage())
            storage = Edge();
        
        NodeType op = GetArrayLength;
        switch (m_node->arrayMode().type()) {
        case Array::ArrayStorage:
        case Array::SlowPutArrayStorage:
            op = GetVectorLength;
            break;
        case Array::String:
            // When we need to support this, it will require additional code since base's useKind is KnownStringUse.
            DFG_CRASH(m_graph, m_node, "Array::String's base.useKind() is KnownStringUse");
            break;
        default:
            break;
        }

        Node* checkInBounds;
#if USE(LARGE_TYPED_ARRAYS)
        if ((op == GetArrayLength) && m_node->arrayMode().isSomeTypedArrayView() && (m_node->arrayMode().mayBeLargeTypedArray() || m_graph.hasExitSite(m_node->origin.semantic, Overflow))) {
            Node* length = m_insertionSet.insertNode(
                m_nodeIndex, SpecInt52Any, GetTypedArrayLengthAsInt52, m_node->origin,
                OpInfo(m_node->arrayMode().asWord()), Edge(base.node(), KnownCellUse), storage);
            if (m_node->arrayMode().mayBeResizableOrGrowableSharedTypedArray())
                m_insertionSet.insertNode(m_nodeIndex, SpecNone, ExitOK, m_node->origin.withExitOK(true));
            // The return type is a dummy since this node does not actually return anything.
            checkInBounds = m_insertionSet.insertNode(
                m_nodeIndex, SpecInt32Only, CheckInBoundsInt52, m_node->origin,
                index, Edge(length, Int52RepUse));
        } else {
#endif
            Node* length = m_insertionSet.insertNode(
                m_nodeIndex, SpecInt32Only, op, m_node->origin,
                OpInfo(m_node->arrayMode().asWord()), Edge(base.node(), KnownCellUse), storage);
            if (m_node->arrayMode().mayBeResizableOrGrowableSharedTypedArray())
                m_insertionSet.insertNode(m_nodeIndex, SpecNone, ExitOK, m_node->origin.withExitOK(true));
            checkInBounds = m_insertionSet.insertNode(
                m_nodeIndex, SpecInt32Only, CheckInBounds, m_node->origin,
                index, Edge(length, KnownInt32Use));
#if USE(LARGE_TYPED_ARRAYS)
        }
#endif


        AdjacencyList adjacencyList = m_graph.copyVarargChildren(m_node);
        m_graph.m_varArgChildren.append(Edge(checkInBounds, UntypedUse));
        adjacencyList.setNumChildren(adjacencyList.numChildren() + 1);
        m_node->children = adjacencyList;
        return true;
    }
    
    InsertionSet m_insertionSet;
    BasicBlock* m_block;
    unsigned m_nodeIndex;
    Node* m_node;
};

bool performSSALowering(Graph& graph)
{
    return runPhase<SSALoweringPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

