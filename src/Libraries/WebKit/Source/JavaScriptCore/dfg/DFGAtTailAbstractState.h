/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#pragma once

#if ENABLE(DFG_JIT)

#include "DFGAbstractInterpreterClobberState.h"
#include "DFGAbstractValue.h"
#include "DFGBasicBlock.h"
#include "DFGBlockMap.h"
#include "DFGGraph.h"
#include "DFGNodeFlowProjection.h"

namespace JSC { namespace DFG { 

class AtTailAbstractState {
public:
    AtTailAbstractState(Graph&);
    
    ~AtTailAbstractState();
    
    explicit operator bool() const { return true; }
    
    void initializeTo(BasicBlock* block)
    {
        m_block = block;
    }
    
    void createValueForNode(NodeFlowProjection);
    
    AbstractValue& fastForward(AbstractValue& value) { return value; }
    
    AbstractValue& forNode(NodeFlowProjection);
    AbstractValue& forNode(Edge edge)
    {
        ASSERT(!edge.node()->isTuple());
        return forNode(edge.node());
    }
    
    ALWAYS_INLINE AbstractValue& forNodeWithoutFastForward(NodeFlowProjection node)
    {
        ASSERT(!node->isTuple());
        return forNode(node);
    }
    
    ALWAYS_INLINE AbstractValue& forNodeWithoutFastForward(Edge edge)
    {
        return forNode(edge);
    }
    
    ALWAYS_INLINE void fastForwardAndFilterUnproven(AbstractValue& value, SpeculatedType type)
    {
        value.filter(type);
    }
    
    ALWAYS_INLINE void clearForNode(NodeFlowProjection node)
    {
        ASSERT(!node->isTuple());
        forNode(node).clear();
    }
    
    ALWAYS_INLINE void clearForNode(Edge edge)
    {
        clearForNode(edge.node());
    }
    
    template<typename... Arguments>
    ALWAYS_INLINE void setForNode(NodeFlowProjection node, Arguments&&... arguments)
    {
        ASSERT(!node->isTuple());
        forNode(node).set(m_graph, std::forward<Arguments>(arguments)...);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setForNode(Edge edge, Arguments&&... arguments)
    {
        setForNode(edge.node(), std::forward<Arguments>(arguments)...);
    }
    
    template<typename... Arguments>
    ALWAYS_INLINE void setTypeForNode(NodeFlowProjection node, Arguments&&... arguments)
    {
        forNode(node).setType(m_graph, std::forward<Arguments>(arguments)...);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setTypeForNode(Edge edge, Arguments&&... arguments)
    {
        setTypeForNode(edge.node(), std::forward<Arguments>(arguments)...);
    }
    
    template<typename... Arguments>
    ALWAYS_INLINE void setNonCellTypeForNode(NodeFlowProjection node, Arguments&&... arguments)
    {
        forNode(node).setNonCellType(std::forward<Arguments>(arguments)...);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setNonCellTypeForNode(Edge edge, Arguments&&... arguments)
    {
        setNonCellTypeForNode(edge.node(), std::forward<Arguments>(arguments)...);
    }
    
    ALWAYS_INLINE void makeBytecodeTopForNode(NodeFlowProjection node)
    {
        forNode(node).makeBytecodeTop();
    }
    
    ALWAYS_INLINE void makeBytecodeTopForNode(Edge edge)
    {
        makeBytecodeTopForNode(edge.node());
    }
    
    ALWAYS_INLINE void makeHeapTopForNode(NodeFlowProjection node)
    {
        forNode(node).makeHeapTop();
    }
    
    ALWAYS_INLINE void makeHeapTopForNode(Edge edge)
    {
        makeHeapTopForNode(edge.node());
    }

    ALWAYS_INLINE AbstractValue& forTupleNodeWithoutFastForward(NodeFlowProjection node, unsigned index)
    {
        return forTupleNode(node, index);
    }

    ALWAYS_INLINE AbstractValue& forTupleNode(NodeFlowProjection node, unsigned index)
    {
        ASSERT(index < node->tupleSize());
        return m_tupleAbstractValues.at(m_block).at(node->tupleOffset() + index);
    }

    ALWAYS_INLINE AbstractValue& forTupleNode(Edge edge, unsigned index)
    {
        return forTupleNode(edge.node(), index);
    }

    ALWAYS_INLINE void clearForTupleNode(NodeFlowProjection node, unsigned index)
    {
        forTupleNode(node, index).clear();
    }

    ALWAYS_INLINE void clearForTupleNode(Edge edge, unsigned index)
    {
        clearForTupleNode(edge.node(), index);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setForTupleNode(NodeFlowProjection node, unsigned index, Arguments&&... arguments)
    {
        forTupleNode(node, index).set(m_graph, std::forward<Arguments>(arguments)...);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setForTupleNode(Edge edge, unsigned index, Arguments&&... arguments)
    {
        setForTupleNode(edge.node(), index, std::forward<Arguments>(arguments)...);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setTypeForTupleNode(NodeFlowProjection node, unsigned index, Arguments&&... arguments)
    {
        forTupleNode(node, index).setType(m_graph, std::forward<Arguments>(arguments)...);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setTypeForTupleNode(Edge edge, unsigned index, Arguments&&... arguments)
    {
        setTypeForTupleNode(edge.node(), index, std::forward<Arguments>(arguments)...);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setNonCellTypeForTupleNode(NodeFlowProjection node, unsigned index, Arguments&&... arguments)
    {
        forTupleNode(node, index).setNonCellType(std::forward<Arguments>(arguments)...);
    }

    template<typename... Arguments>
    ALWAYS_INLINE void setNonCellTypeForTupleNode(Edge edge, unsigned index, Arguments&&... arguments)
    {
        setNonCellTypeForTupleNode(edge.node(), index, std::forward<Arguments>(arguments)...);
    }

    ALWAYS_INLINE void makeBytecodeTopForTupleNode(NodeFlowProjection node, unsigned index)
    {
        forTupleNode(node, index).makeBytecodeTop();
    }

    ALWAYS_INLINE void makeBytecodeTopForTupleNode(Edge edge, unsigned index)
    {
        makeBytecodeTopForTupleNode(edge.node(), index);
    }

    ALWAYS_INLINE void makeHeapTopForTupleNode(NodeFlowProjection node, unsigned index)
    {
        forTupleNode(node, index).makeHeapTop();
    }

    ALWAYS_INLINE void makeHeapTopForTupleNode(Edge edge, unsigned index)
    {
        makeHeapTopForTupleNode(edge.node(), index);
    }
    
    unsigned size() const { return m_block->valuesAtTail.size(); }
    unsigned numberOfArguments() const { return m_block->valuesAtTail.numberOfArguments(); }
    unsigned numberOfLocals() const { return m_block->valuesAtTail.numberOfLocals(); }
    unsigned numberOfTmps() const { return m_block->valuesAtTail.numberOfTmps(); }
    AbstractValue& atIndex(size_t index) { return m_block->valuesAtTail.at(index); }
    AbstractValue& operand(Operand operand) { return m_block->valuesAtTail.operand(operand); }
    AbstractValue& local(size_t index) { return m_block->valuesAtTail.local(index); }
    AbstractValue& argument(size_t index) { return m_block->valuesAtTail.argument(index); }
    AbstractValue& tmp(size_t index) { return m_block->valuesAtTail.tmp(index); }
    
    void clobberStructures()
    {
        UNREACHABLE_FOR_PLATFORM();
    }
    
    void observeInvalidationPoint()
    {
        UNREACHABLE_FOR_PLATFORM();
    }
    
    BasicBlock* block() const { return m_block; }
    
    bool isValid() { return m_block->cfaDidFinish; }
    
    StructureClobberState structureClobberState() const { return m_block->cfaStructureClobberStateAtTail; }
    
    void setClobberState(AbstractInterpreterClobberState) { }
    void mergeClobberState(AbstractInterpreterClobberState) { }
    void setStructureClobberState(StructureClobberState state) { RELEASE_ASSERT(state == m_block->cfaStructureClobberStateAtTail); }
    void setIsValid(bool isValid) { m_block->cfaDidFinish = isValid; }
    void setBranchDirection(BranchDirection) { }
    void setShouldTryConstantFolding(bool) { }

    void trustEdgeProofs() { m_trustEdgeProofs = true; }
    void dontTrustEdgeProofs() { m_trustEdgeProofs = false; }
    void setProofStatus(Edge& edge, ProofStatus status)
    {
        if (m_trustEdgeProofs)
            edge.setProofStatus(status);
    }

private:
    Graph& m_graph;
    BlockMap<UncheckedKeyHashMap<NodeFlowProjection, AbstractValue>> m_valuesAtTailMap;
    BlockMap<Vector<AbstractValue>> m_tupleAbstractValues;
    BasicBlock* m_block { nullptr };
    bool m_trustEdgeProofs { false };
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
