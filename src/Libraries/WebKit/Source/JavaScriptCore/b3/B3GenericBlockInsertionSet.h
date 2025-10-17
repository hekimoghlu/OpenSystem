/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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

#if ENABLE(B3_JIT)

#include "PureNaN.h"
#include <climits>
#include <wtf/BubbleSort.h>
#include <wtf/Insertion.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

class InsertionSet;

template<typename BasicBlock>
class GenericBlockInsertionSet {
public:
    typedef WTF::Insertion<std::unique_ptr<BasicBlock>> BlockInsertion;
    
    GenericBlockInsertionSet(Vector<std::unique_ptr<BasicBlock>>& blocks)
        : m_blocks(blocks)
    {
    }
    
    void insert(BlockInsertion&& insertion)
    {
        m_insertions.append(WTFMove(insertion));
    }

    // Insert a new block at a given index.
    BasicBlock* insert(unsigned index, double frequency = PNaN)
    {
        std::unique_ptr<BasicBlock> block(new BasicBlock(BasicBlock::uninsertedIndex, frequency));
        BasicBlock* result = block.get();
        insert(BlockInsertion(index, WTFMove(block)));
        return result;
    }

    // Inserts a new block before the given block. Usually you will not pass the frequency
    // argument. Passing PNaN causes us to just use the frequency of the 'before' block. That's
    // usually what you want.
    BasicBlock* insertBefore(BasicBlock* before, double frequency = PNaN)
    {
        return insert(before->index(), frequency == frequency ? frequency : before->frequency());
    }

    // Inserts a new block after the given block.
    BasicBlock* insertAfter(BasicBlock* after, double frequency = PNaN)
    {
        return insert(after->index() + 1, frequency == frequency ? frequency : after->frequency());
    }

    bool execute()
    {
        if (m_insertions.isEmpty())
            return false;
        
        // We allow insertions to be given to us in any order. So, we need to sort them before
        // running WTF::executeInsertions. We strongly prefer a stable sort and we want it to be
        // fast, so we use bubble sort.
        bubbleSort(m_insertions.begin(), m_insertions.end());
        
        executeInsertions(m_blocks, m_insertions);
        
        // Prune out empty entries. This isn't strictly necessary but it's
        // healthy to keep the block list from growing.
        m_blocks.removeAllMatching(
            [&] (std::unique_ptr<BasicBlock>& blockPtr) -> bool {
                return !blockPtr;
            });
        
        // Make sure that the blocks know their new indices.
        for (unsigned i = 0; i < m_blocks.size(); ++i)
            m_blocks[i]->m_index = i;
        
        return true;
    }

private:
    Vector<std::unique_ptr<BasicBlock>>& m_blocks;
    Vector<BlockInsertion, 8> m_insertions;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
