/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#include <wtf/BitVector.h>

namespace JSC { namespace DFG {

class Graph;

class BlockSet {
public:
    BlockSet() { }
    
    // Return true if the block was added, false if it was already present.
    bool add(BasicBlock* block)
    {
        return !m_set.set(block->index);
    }
    
    // Return true if the block was removed, false if it was already absent.
    bool remove(BasicBlock* block)
    {
        return m_set.clear(block->index);
    }
    
    bool contains(BasicBlock* block) const
    {
        if (!block)
            return false;
        return m_set.get(block->index);
    }
    
    class iterator {
    public:
        iterator()
            : m_graph(nullptr)
            , m_set(nullptr)
            , m_index(0)
        {
        }
        
        iterator& operator++()
        {
            m_index = m_set->m_set.findBit(m_index + 1, true);
            return *this;
        }
        
        BasicBlock* operator*() const;
        
        bool operator==(const iterator& other) const
        {
            return m_index == other.m_index;
        }
        
    private:
        friend class BlockSet;
        
        Graph* m_graph;
        const BlockSet* m_set;
        size_t m_index;
    };
    
    class Iterable {
    public:
        Iterable(Graph& graph, const BlockSet& set)
            : m_graph(graph)
            , m_set(set)
        {
        }
        
        iterator begin() const
        {
            iterator result;
            result.m_graph = &m_graph;
            result.m_set = &m_set;
            result.m_index = m_set.m_set.findBit(0, true);
            return result;
        }
        
        iterator end() const
        {
            iterator result;
            result.m_graph = &m_graph;
            result.m_set = &m_set;
            result.m_index = m_set.m_set.size();
            return result;
        }
        
    private:
        Graph& m_graph;
        const BlockSet& m_set;
    };
    
    Iterable iterable(Graph& graph) const
    {
        return Iterable(graph, *this);
    }
    
    void dump(PrintStream&) const;
    
private:
    BitVector m_set;
};

class BlockAdder {
public:
    BlockAdder(BlockSet& set)
        : m_set(set)
    {
    }
    
    bool operator()(BasicBlock* block) const
    {
        return m_set.add(block);
    }
private:
    BlockSet& m_set;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
