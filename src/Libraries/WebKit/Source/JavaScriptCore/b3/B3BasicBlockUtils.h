/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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

#include <wtf/GraphNodeWorklist.h>
#include <wtf/IndexSet.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

template<typename BasicBlock>
bool addPredecessor(BasicBlock* block, BasicBlock* predecessor)
{
    auto& predecessors = block->predecessors();

    if (predecessors.contains(predecessor))
        return false;

    predecessors.append(predecessor);
    return true;
}

template<typename BasicBlock>
bool removePredecessor(BasicBlock* block, BasicBlock* predecessor)
{
    auto& predecessors = block->predecessors();
    for (unsigned i = 0; i < predecessors.size(); ++i) {
        if (predecessors[i] == predecessor) {
            predecessors[i--] = predecessors.last();
            predecessors.removeLast();
            ASSERT(!predecessors.contains(predecessor));
            return true;
        }
    }
    return false;
}

template<typename BasicBlock>
bool replacePredecessor(BasicBlock* block, BasicBlock* from, BasicBlock* to)
{
    bool changed = false;
    // We do it this way because 'to' may already be a predecessor of 'block'.
    changed |= removePredecessor(block, from);
    changed |= addPredecessor(block, to);
    return changed;
}

template<typename BasicBlock>
void updatePredecessorsAfter(BasicBlock* root)
{
    Vector<BasicBlock*, 16> worklist;
    worklist.append(root);
    while (!worklist.isEmpty()) {
        BasicBlock* block = worklist.takeLast();
        for (BasicBlock* successor : block->successorBlocks()) {
            if (addPredecessor(successor, block))
                worklist.append(successor);
        }
    }
}

template<typename BasicBlock>
void clearPredecessors(Vector<std::unique_ptr<BasicBlock>>& blocks)
{
    for (auto& block : blocks) {
        if (block)
            block->predecessors().shrink(0);
    }
}

template<typename BasicBlock>
void recomputePredecessors(Vector<std::unique_ptr<BasicBlock>>& blocks)
{
    clearPredecessors(blocks);
    updatePredecessorsAfter(blocks[0].get());
}

template<typename BasicBlock>
bool isBlockDead(BasicBlock* block)
{
    if (!block)
        return false;
    if (!block->index())
        return false;
    return block->predecessors().isEmpty();
}

template<typename BasicBlock>
Vector<BasicBlock*> blocksInPreOrder(BasicBlock* root)
{
    Vector<BasicBlock*> result;
    GraphNodeWorklist<BasicBlock*, IndexSet<BasicBlock*>> worklist;
    worklist.push(root);
    while (BasicBlock* block = worklist.pop()) {
        result.append(block);
        for (BasicBlock* successor : block->successorBlocks())
            worklist.push(successor);
    }
    return result;
}

template<typename BasicBlock>
Vector<BasicBlock*> blocksInPostOrder(BasicBlock* root)
{
    Vector<BasicBlock*> result;
    PostOrderGraphNodeWorklist<BasicBlock*, IndexSet<BasicBlock*>> worklist;
    worklist.push(root);
    while (GraphNodeWithOrder<BasicBlock*> item = worklist.pop()) {
        switch (item.order) {
        case GraphVisitOrder::Pre:
            worklist.pushPost(item.node);
            for (BasicBlock* successor : item.node->successorBlocks())
                worklist.push(successor);
            break;
        case GraphVisitOrder::Post:
            result.append(item.node);
            break;
        }
    }
    return result;
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
