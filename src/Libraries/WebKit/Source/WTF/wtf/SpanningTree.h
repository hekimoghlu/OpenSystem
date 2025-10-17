/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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

#include <wtf/GraphNodeWorklist.h>

template<typename Graph>
class SpanningTree {
    WTF_MAKE_FAST_ALLOCATED;
public:
    SpanningTree(Graph& graph)
        : m_graph(graph)
        , m_data(graph.template newMap<Data>())
    {
        ExtendedGraphNodeWorklist<typename Graph::Node, unsigned, typename Graph::Set> worklist;
        worklist.push(m_graph.root(), 0);
    
        size_t number = 0;

        while (GraphNodeWith<typename Graph::Node, unsigned> item = worklist.pop()) {
            typename Graph::Node block = item.node;
            unsigned successorIndex = item.data;
        
            // We initially push with successorIndex = 0 regardless of whether or not we have any
            // successors. This is so that we can assign our prenumber. Subsequently we get pushed
            // with higher successorIndex values. We finally push successorIndex == # successors
            // to calculate our post number.
            ASSERT(!successorIndex || successorIndex <= m_graph.successors(block).size());

            if (!successorIndex)
                m_data[block].pre = number++;
        
            if (successorIndex < m_graph.successors(block).size()) {
                unsigned nextSuccessorIndex = successorIndex + 1;
                // We need to push this even if this is out of bounds so we can compute
                // the post number.
                worklist.forcePush(block, nextSuccessorIndex);

                typename Graph::Node successorBlock = m_graph.successors(block)[successorIndex];
                worklist.push(successorBlock, 0);
            } else
                m_data[block].post = number++;
        }
    }

    // Returns true if a is a descendent of b.
    // Note a is a descendent of b if they're equal.
    bool isDescendent(typename Graph::Node a, typename Graph::Node b)
    {
        return m_data[b].pre <= m_data[a].pre
            && m_data[b].post >= m_data[a].post;
    }

private:
    struct Data {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        size_t pre;
        size_t post;
    };

    Graph& m_graph;
    typename Graph::template Map<Data> m_data;
};
