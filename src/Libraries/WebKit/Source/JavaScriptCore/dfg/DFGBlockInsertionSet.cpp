/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
#include "DFGBlockInsertionSet.h"

#if ENABLE(DFG_JIT)

#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

BlockInsertionSet::BlockInsertionSet(Graph& graph)
    : m_graph(graph)
{
}

BlockInsertionSet::~BlockInsertionSet() = default;

void BlockInsertionSet::insert(BlockInsertion&& insertion)
{
    m_insertions.append(WTFMove(insertion));
}

void BlockInsertionSet::insert(size_t index, std::unique_ptr<BasicBlock>&& block)
{
    insert(BlockInsertion(index, WTFMove(block)));
}

BasicBlock* BlockInsertionSet::insert(size_t index, float executionCount)
{
    auto block = makeUnique<BasicBlock>(BytecodeIndex(), m_graph.block(0)->variablesAtHead.numberOfArguments(), m_graph.block(0)->variablesAtHead.numberOfLocals(), m_graph.block(0)->variablesAtHead.numberOfTmps(), executionCount);
    block->isReachable = true;
    auto* result = block.get();
    insert(index, WTFMove(block));
    return result;
}

BasicBlock* BlockInsertionSet::insertBefore(BasicBlock* before, float executionCount)
{
    return insert(before->index, executionCount);
}

bool BlockInsertionSet::execute()
{
    if (m_insertions.isEmpty())
        return false;
    
    // We allow insertions to be given to us in any order. So, we need to sort them before
    // running WTF::executeInsertions. Also, we don't really care if the sort is stable since
    // basic block order doesn't have semantics - it's just to make code easier to read.
    std::sort(m_insertions.begin(), m_insertions.end());

    executeInsertions(m_graph.m_blocks, m_insertions);
    
    // Prune out empty entries. This isn't strictly necessary but it's
    // healthy to keep the block list from growing.
    unsigned targetIndex = 0;
    for (unsigned sourceIndex = 0; sourceIndex < m_graph.m_blocks.size();) {
        auto block = WTFMove(m_graph.m_blocks[sourceIndex++]);
        if (!block)
            continue;
        m_graph.m_blocks[targetIndex++] = WTFMove(block);
    }
    m_graph.m_blocks.shrink(targetIndex);
    
    // Make sure that the blocks know their new indices.
    for (unsigned i = 0; i < m_graph.m_blocks.size(); ++i)
        m_graph.m_blocks[i]->index = i;
    
    // And finally, invalidate all analyses that rely on the CFG.
    m_graph.invalidateCFG();
    m_graph.dethread();
    
    return true;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

