/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#include "DFGNode.h"

namespace JSC { namespace DFG {

class PureValue {
public:
    PureValue()
        : m_op(LastNodeType)
        , m_info(0)
    {
    }
    
    PureValue(NodeType op, const AdjacencyList& children, uintptr_t info)
        : m_op(op)
        , m_children(children.sanitized())
        , m_info(info)
    {
        ASSERT(!(defaultFlags(op) & NodeHasVarArgs));
    }
    
    PureValue(NodeType op, const AdjacencyList& children, const void* ptr)
        : PureValue(op, children, std::bit_cast<uintptr_t>(ptr))
    {
    }
    
    PureValue(NodeType op, const AdjacencyList& children)
        : PureValue(op, children, static_cast<uintptr_t>(0))
    {
    }
    
    PureValue(Node* node, uintptr_t info)
        : PureValue(node->op(), node->children, info)
    {
    }
    
    PureValue(Node* node, const void* ptr)
        : PureValue(node->op(), node->children, ptr)
    {
    }
    
    PureValue(Node* node)
        : PureValue(node->op(), node->children)
    {
    }

    PureValue(Graph& graph, Node* node, uintptr_t info)
        : m_op(node->op())
        , m_children(node->children)
        , m_info(info)
        , m_graph(&graph)
    {
        ASSERT(node->flags() & NodeHasVarArgs);
        ASSERT(isVarargs());
    }

    PureValue(Graph& graph, Node* node)
        : PureValue(graph, node, static_cast<uintptr_t>(0))
    {
    }

    PureValue(WTF::HashTableDeletedValueType)
        : m_op(LastNodeType)
        , m_info(1)
    {
    }
    
    bool operator!() const { return m_op == LastNodeType && !m_info; }
    
    NodeType op() const { return m_op; }
    uintptr_t info() const { return m_info; }

    unsigned hash() const
    {
        unsigned hash = WTF::IntHash<int>::hash(static_cast<int>(m_op)) + m_info;
        if (!isVarargs())
            return hash ^ m_children.hash();
        for (unsigned i = 0; i < m_children.numChildren(); ++i)
            hash ^= m_graph->m_varArgChildren[m_children.firstChild() + i].sanitized().hash();
        return hash;
    }
    
    bool operator==(const PureValue& other) const
    {
        if (isVarargs() != other.isVarargs() || m_op != other.m_op || m_info != other.m_info)
            return false;
        if (!isVarargs())
            return m_children == other.m_children;
        if (m_children.numChildren() != other.m_children.numChildren())
            return false;
        for (unsigned i = 0; i < m_children.numChildren(); ++i) {
            Edge a = m_graph->m_varArgChildren[m_children.firstChild() + i].sanitized();
            Edge b = m_graph->m_varArgChildren[other.m_children.firstChild() + i].sanitized();
            if (a != b)
                return false;
        }
        return true;
    }
    
    bool isHashTableDeletedValue() const
    {
        return m_op == LastNodeType && m_info;
    }
    
    void dump(PrintStream& out) const;
    
private:
    bool isVarargs() const { return !!m_graph; }

    NodeType m_op;
    AdjacencyList m_children;
    uintptr_t m_info;
    Graph* m_graph { nullptr };
};

struct PureValueHash {
    static unsigned hash(const PureValue& key) { return key.hash(); }
    static bool equal(const PureValue& a, const PureValue& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } // namespace JSC::DFG

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::DFG::PureValue> : JSC::DFG::PureValueHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::DFG::PureValue> : SimpleClassHashTraits<JSC::DFG::PureValue> {
    static constexpr bool emptyValueIsZero = false;
};

} // namespace WTF

namespace JSC { namespace DFG {

typedef UncheckedKeyHashMap<PureValue, Node*> PureMap;
typedef UncheckedKeyHashMap<PureValue, Vector<Node*>> PureMultiMap;

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
