/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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

#include "DFGGraph.h"
#include <wtf/Insertion.h>
#include <wtf/Vector.h>

namespace JSC { namespace DFG {

typedef WTF::Insertion<Node*> Insertion;

class InsertionSet {
public:
    InsertionSet(Graph& graph)
        : m_graph(graph)
    {
    }
    
    Graph& graph() { return m_graph; }

    // Adds another code insertion. It's expected that you'll usually insert things in order. If
    // you don't, this function will perform a linear search to find the largest insertion point
    // at which insertion order would be preserved. This is essentially equivalent to if you did
    // a stable sort on the insertions.
    Node* insert(const Insertion& insertion)
    {
        if (LIKELY(!m_insertions.size() || m_insertions.last().index() <= insertion.index()))
            m_insertions.append(insertion);
        else
            insertSlow(insertion);
        return insertion.element();
    }
    
    Node* insert(size_t index, Node* element)
    {
        return insert(Insertion(index, element));
    }

    template<typename... Params>
    Node* insertNode(size_t index, SpeculatedType type, Params... params)
    {
        return insert(index, m_graph.addNode(type, params...));
    }
    
    Node* insertConstant(size_t index, NodeOrigin, FrozenValue*, NodeType op = JSConstant);

    Edge insertConstantForUse(
        size_t index, NodeOrigin origin, FrozenValue* value, UseKind useKind)
    {
        NodeType op;
        if (isDouble(useKind))
            op = DoubleConstant;
        else if (useKind == Int52RepUse)
            op = Int52Constant;
        else
            op = JSConstant;
        return Edge(insertConstant(index, origin, value, op), useKind);
    }
    
    Node* insertConstant(size_t index, NodeOrigin origin, JSValue value, NodeType op = JSConstant)
    {
        return insertConstant(index, origin, m_graph.freeze(value), op);
    }
    
    Edge insertConstantForUse(size_t index, NodeOrigin origin, JSValue value, UseKind useKind)
    {
        return insertConstantForUse(index, origin, m_graph.freeze(value), useKind);
    }
    
    Edge insertBottomConstantForUse(size_t index, NodeOrigin origin, UseKind useKind)
    {
        if (isDouble(useKind))
            return insertConstantForUse(index, origin, jsNumber(PNaN), useKind);
        if (useKind == Int52RepUse)
            return insertConstantForUse(index, origin, jsNumber(0), useKind);
        return insertConstantForUse(index, origin, jsUndefined(), useKind);
    }
    
    Node* insertCheck(size_t index, NodeOrigin origin, AdjacencyList children)
    {
        children = children.justChecks();
        if (children.isEmpty())
            return nullptr;
        return insertNode(index, SpecNone, Check, origin, children);
    }
    
    Node* insertCheck(Graph& graph, size_t index, Node* node)
    {
        if (!(node->flags() & NodeHasVarArgs))
            return insertCheck(index, node->origin, node->children);

        AdjacencyList children = graph.copyVarargChildren(node, [] (Edge edge) { return edge.willHaveCheck(); });
        if (!children.numChildren())
            return nullptr;
        return insertNode(index, SpecNone, CheckVarargs, node->origin, children);
    }
    
    Node* insertCheck(size_t index, NodeOrigin origin, Edge edge)
    {
        if (edge.willHaveCheck())
            return insertNode(index, SpecNone, Check, origin, edge);
        return nullptr;
    }
    
    size_t execute(BasicBlock* block);

private:
    void insertSlow(const Insertion&);
    
    Graph& m_graph;
    Vector<Insertion, 8> m_insertions;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
