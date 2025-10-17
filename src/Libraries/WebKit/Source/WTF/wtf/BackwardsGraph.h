/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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

#include <wtf/FastMalloc.h>
#include <wtf/GraphNodeWorklist.h>
#include <wtf/Noncopyable.h>
#include <wtf/SingleRootGraph.h>
#include <wtf/SpanningTree.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

template<typename Graph>
class BackwardsGraph {
    WTF_MAKE_NONCOPYABLE(BackwardsGraph);
    WTF_MAKE_FAST_ALLOCATED;
public:
    using Node = SingleRootGraphNode<Graph>;
    using Set = SingleRootGraphSet<Graph>;
    template <typename T> using Map = SingleRootMap<T, Graph>;
    typedef Vector<Node, 4> List;

    BackwardsGraph(Graph& graph)
        : m_graph(graph)
    {
        GraphNodeWorklist<typename Graph::Node, typename Graph::Set> worklist;

        auto addRootSuccessor = [&] (typename Graph::Node node) {
            if (worklist.push(node)) {
                m_rootSuccessorList.append(node);
                m_rootSuccessorSet.add(node);
                while (typename Graph::Node node = worklist.pop())
                    worklist.pushAll(graph.predecessors(node));
            }
        };

        {
            // Loops are a form of terminality (you can loop forever). To have a loop, you need to
            // have a back edge. An edge u->v is a back edge when u is a descendent of v in the
            // DFS spanning tree of the Graph.
            SpanningTree<Graph> spanningTree(graph);
            for (unsigned i = 0; i < graph.numNodes(); ++i) {
                if (typename Graph::Node node = graph.node(i)) {
                    for (typename Graph::Node successor : graph.successors(node)) {
                        if (spanningTree.isDescendent(node, successor)) {
                            addRootSuccessor(node);
                            break;
                        }
                    }
                }
            }
        }

        for (unsigned i = 0; i < graph.numNodes(); ++i) {
            if (typename Graph::Node node = graph.node(i)) {
                if (!graph.successors(node).size())
                    addRootSuccessor(node);
            }
        }

        // At this point there will be some nodes in the graph that aren't known to the worklist. We
        // could add any or all of them to the root successors list. Adding all of them would be a bad
        // pessimisation. Ideally we would pick the ones that have backward edges but no forward
        // edges. That would require thinking, so we just use a rough heuristic: add the highest
        // numbered nodes first, which is totally fine if the input program is already sorted nicely.
        for (unsigned i = graph.numNodes(); i--;) {
            if (typename Graph::Node node = graph.node(i))
                addRootSuccessor(node);
        }
    }

    Node root() { return Node::root(); }

    template<typename T>
    Map<T> newMap() { return Map<T>(m_graph); }

    List successors(const Node& node) const
    {
        if (node.isRoot())
            return m_rootSuccessorList;
        List result;
        for (typename Graph::Node predecessor : m_graph.predecessors(node.node()))
            result.append(predecessor);
        return result;
    }

    List predecessors(const Node& node) const
    {
        if (node.isRoot())
            return { };

        List result;
        
        if (m_rootSuccessorSet.contains(node.node()))
            result.append(Node::root());

        for (typename Graph::Node successor : m_graph.successors(node.node()))
            result.append(successor);

        return result;
    }

    unsigned index(const Node& node) const
    {
        if (node.isRoot())
            return 0;
        return m_graph.index(node.node()) + 1;
    }

    Node node(unsigned index) const
    {
        if (!index)
            return Node::root();
        return m_graph.node(index - 1);
    }

    unsigned numNodes() const
    {
        return m_graph.numNodes() + 1;
    }

    CString dump(Node node) const
    {
        StringPrintStream out;
        if (!node)
            out.print("<null>");
        else if (node.isRoot())
            out.print(Node::rootName());
        else
            out.print(m_graph.dump(node.node()));
        return out.toCString();
    }

    void dump(PrintStream& out) const
    {
        for (unsigned i = 0; i < numNodes(); ++i) {
            Node node = this->node(i);
            if (!node)
                continue;
            out.print(dump(node), ":\n"_s);
            out.print("    Preds: "_s);
            CommaPrinter comma;
            for (Node predecessor : predecessors(node))
                out.print(comma, dump(predecessor));
            out.print("\n"_s);
            out.print("    Succs: "_s);
            comma = CommaPrinter();
            for (Node successor : successors(node))
                out.print(comma, dump(successor));
            out.print("\n"_s);
        }
    }

private:
    Graph& m_graph;
    List m_rootSuccessorList;
    typename Graph::Set m_rootSuccessorSet;
};

} // namespace WTF

using WTF::BackwardsGraph;
