/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#include "DFGBasicBlock.h"
#include "DFGBlockMapInlines.h"
#include "DFGBlockSet.h"
#include "DFGGraph.h"
#include <wtf/SingleRootGraph.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace DFG {

class CFG {
    WTF_MAKE_NONCOPYABLE(CFG);
    WTF_MAKE_TZONE_ALLOCATED(CFG);
public:
    typedef BasicBlock* Node;
    typedef BlockSet Set;
    template<typename T> using Map = BlockMap<T>;
    typedef BlockList List;

    CFG(Graph& graph)
        : m_graph(graph)
    {
    }

    Node root()
    {
        ASSERT(m_graph.m_form == SSA || m_graph.m_isInSSAConversion);
        return m_graph.block(0);
    }

    List roots()
    {
        List result;
        for (BasicBlock* root : m_graph.m_roots)
            result.append(root);
        return result;
    }

    template<typename T>
    Map<T> newMap() { return BlockMap<T>(m_graph); }

    DFG::Node::SuccessorsIterable successors(Node node) { return node->successors(); }
    PredecessorList& predecessors(Node node) { return node->predecessors; }

    unsigned index(Node node) const { return node->index; }
    Node node(unsigned index) const { return m_graph.block(index); }
    unsigned numNodes() const { return m_graph.numBlocks(); }
    
    PointerDump<BasicBlock> dump(Node node) const { return pointerDump(node); }

    void dump(PrintStream& out) const
    {
        m_graph.dump(out);
    }

private:
    Graph& m_graph;
};

class CPSCFG : public SingleRootGraph<CFG> {
public:
    CPSCFG(Graph& graph)
        : SingleRootGraph<CFG>(*graph.m_ssaCFG)
    {
        ASSERT(graph.m_roots.size());
    }
};

using SSACFG = CFG;

template <typename T, typename = typename std::enable_if<std::is_same<T, CPSCFG>::value>::type>
CPSCFG& selectCFG(Graph& graph)
{
    return graph.ensureCPSCFG();
}

template <typename T, typename = typename std::enable_if<std::is_same<T, SSACFG>::value>::type>
SSACFG& selectCFG(Graph& graph)
{
    RELEASE_ASSERT(graph.m_ssaCFG);
    return *graph.m_ssaCFG;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
