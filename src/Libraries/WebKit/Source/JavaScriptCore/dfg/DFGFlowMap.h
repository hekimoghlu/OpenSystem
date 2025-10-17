/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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

#include "DFGFlowIndexing.h"
#include "DFGGraph.h"
#include "DFGNode.h"
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace DFG {

// This is a mapping from nodes to values that is useful for flow-sensitive analysis. In such an
// analysis, at every point in the program we need to consider the values of nodes plus the shadow
// values of Phis. This makes it easy to do both of those things.
template<typename T>
class FlowMap {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(FlowMap);
public:
    FlowMap(Graph& graph)
        : m_graph(graph)
    {
        resize();
    }
    
    // Call this if the number of nodes in the graph has changed. Note that this does not reset any
    // entries.
    void resize()
    {
        m_map.resize(m_graph.maxNodeCount());
        if (m_graph.m_form == SSA)
            m_shadowMap.resize(m_graph.maxNodeCount());
    }
    
    Graph& graph() const { return m_graph; }
    
    ALWAYS_INLINE T& at(unsigned nodeIndex)
    {
        return m_map[nodeIndex];
    }
    
    ALWAYS_INLINE T& at(Node* node)
    {
        return at(node->index());
    }
    
    ALWAYS_INLINE T& atShadow(unsigned nodeIndex)
    {
        return m_shadowMap[nodeIndex];
    }
    
    ALWAYS_INLINE T& atShadow(Node* node)
    {
        return atShadow(node->index());
    }
    
    ALWAYS_INLINE T& at(unsigned nodeIndex, NodeFlowProjection::Kind kind)
    {
        switch (kind) {
        case NodeFlowProjection::Primary:
            return at(nodeIndex);
        case NodeFlowProjection::Shadow:
            return atShadow(nodeIndex);
        }
        RELEASE_ASSERT_NOT_REACHED();
        return *std::bit_cast<T*>(nullptr);
    }
    
    ALWAYS_INLINE T& at(Node* node, NodeFlowProjection::Kind kind)
    {
        return at(node->index(), kind);
    }
    
    ALWAYS_INLINE T& at(NodeFlowProjection projection)
    {
        return at(projection.node(), projection.kind());
    }
    
    ALWAYS_INLINE const T& at(unsigned nodeIndex) const { return const_cast<FlowMap*>(this)->at(nodeIndex); }
    ALWAYS_INLINE const T& at(Node* node) const { return const_cast<FlowMap*>(this)->at(node); }
    ALWAYS_INLINE const T& atShadow(unsigned nodeIndex) const { return const_cast<FlowMap*>(this)->atShadow(nodeIndex); }
    ALWAYS_INLINE const T& atShadow(Node* node) const { return const_cast<FlowMap*>(this)->atShadow(node); }
    ALWAYS_INLINE const T& at(unsigned nodeIndex, NodeFlowProjection::Kind kind) const { return const_cast<FlowMap*>(this)->at(nodeIndex, kind); }
    ALWAYS_INLINE const T& at(Node* node, NodeFlowProjection::Kind kind) const { return const_cast<FlowMap*>(this)->at(node, kind); }
    ALWAYS_INLINE const T& at(NodeFlowProjection projection) const { return const_cast<FlowMap*>(this)->at(projection); }

    ALWAYS_INLINE void clear()
    {
        m_map.clear();
        m_shadowMap.clear();
        resize();
    }

private:
    Graph& m_graph;
    Vector<T, 0, UnsafeVectorOverflow> m_map;
    Vector<T, 0, UnsafeVectorOverflow> m_shadowMap;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename T>, FlowMap<T>);

} } // namespace JSC::DFG

namespace WTF {

template<typename T>
void printInternal(PrintStream& out, const JSC::DFG::FlowMap<T>& map)
{
    CommaPrinter comma;
    for (unsigned i = 0; i < map.graph().maxNodeCount(); ++i) {
        if (JSC::DFG::Node* node = map.graph().nodeAt(i)) {
            if (const T& value = map.at(node))
                out.print(comma, node, "=>"_s, value);
        }
    }
    for (unsigned i = 0; i < map.graph().maxNodeCount(); ++i) {
        if (JSC::DFG::Node* node = map.graph().nodeAt(i)) {
            if (const T& value = map.atShadow(node))
                out.print(comma, "shadow("_s, node, ")=>"_s, value);
        }
    }
}

} // namespace WTF

#endif // ENABLE(DFG_JIT)

