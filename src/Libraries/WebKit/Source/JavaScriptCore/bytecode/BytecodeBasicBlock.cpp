/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#include "BytecodeBasicBlock.h"

#include "CodeBlock.h"
#include "PreciseJumpTargets.h"
#include "StrongInlines.h"
#include "UnlinkedCodeBlockGenerator.h"

namespace JSC {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(BytecodeBasicBlock);

template<typename OpcodeTraits>
BytecodeBasicBlock<OpcodeTraits>::BytecodeBasicBlock(const typename InstructionStreamType::Ref& instruction, unsigned blockIndex)
    : m_leaderOffset(instruction.offset())
    , m_totalLength(0)
    , m_index(blockIndex)
{
    addLength(instruction->size());
}

template<typename OpcodeTraits>
BytecodeBasicBlock<OpcodeTraits>::BytecodeBasicBlock(typename BytecodeBasicBlock<OpcodeTraits>::SpecialBlockType blockType, unsigned blockIndex)
    : m_leaderOffset(blockType == BytecodeBasicBlock<OpcodeTraits>::EntryBlock ? 0 : UINT_MAX)
    , m_totalLength(blockType == BytecodeBasicBlock<OpcodeTraits>::EntryBlock ? 0 : UINT_MAX)
    , m_index(blockIndex)
{
}

template<typename OpcodeTraits>
void BytecodeBasicBlock<OpcodeTraits>::addLength(unsigned bytecodeLength)
{
    m_delta.append(bytecodeLength);
    m_totalLength += bytecodeLength;
}

template<typename OpcodeTraits>
void BytecodeBasicBlock<OpcodeTraits>::shrinkToFit()
{
    m_delta.shrinkToFit();
    m_successors.shrinkToFit();
}

template<typename OpcodeTraits>
static bool isJumpTarget(typename OpcodeTraits::OpcodeID opcodeID, const Vector<typename BytecodeBasicBlock<OpcodeTraits>::InstructionStreamType::Offset, 32>& jumpTargets, unsigned bytecodeOffset)
{
    if (opcodeID == op_catch)
        return true;

    return std::binary_search(jumpTargets.begin(), jumpTargets.end(), bytecodeOffset);
}

template<typename OpcodeTraits>
template<typename Block>
auto BytecodeBasicBlock<OpcodeTraits>::computeImpl(Block* codeBlock, const InstructionStreamType& instructions) -> BasicBlockVector
{
    BasicBlockVector basicBlocks;
    Vector<typename InstructionStreamType::Offset, 32> jumpTargets;
    computePreciseJumpTargets(codeBlock, instructions, jumpTargets);

    auto linkBlocks = [&] (BytecodeBasicBlock<OpcodeTraits>& from, BytecodeBasicBlock<OpcodeTraits>& to) {
        from.addSuccessor(to);
    };

    {
        // Create the entry and exit basic blocks.
        basicBlocks.reserveCapacity(jumpTargets.size() + 2);
        {
            // Entry block.
            basicBlocks.constructAndAppend(BytecodeBasicBlock<OpcodeTraits>::EntryBlock, basicBlocks.size());
            // First block.
            basicBlocks.constructAndAppend(BytecodeBasicBlock<OpcodeTraits>::EntryBlock, basicBlocks.size());
            linkBlocks(basicBlocks[0], basicBlocks[1]);
        }

        auto* current = &basicBlocks.last();
        auto appendBlock = [&] (const typename InstructionStreamType::Ref& instruction) -> BytecodeBasicBlock<OpcodeTraits>* {
            basicBlocks.constructAndAppend(instruction, basicBlocks.size());
            return &basicBlocks.last();
        };
        bool nextInstructionIsLeader = false;
        for (const auto& instruction : instructions) {
            auto bytecodeOffset = instruction.offset();
            auto opcodeID = instruction->opcodeID();

            bool createdBlock = false;
            // If the current bytecode is a jump target, then it's the leader of its own basic block.
            if (nextInstructionIsLeader || isJumpTarget<OpcodeTraits>(opcodeID, jumpTargets, bytecodeOffset)) {
                current = appendBlock(instruction);
                createdBlock = true;
                nextInstructionIsLeader = false;
            }

            // If the current bytecode is a branch or a return, then the next instruction is the leader of its own basic block.
            if (isBranch(opcodeID) || isTerminal(opcodeID) || isThrow(opcodeID))
                nextInstructionIsLeader = true;

            if (createdBlock)
                continue;

            // Otherwise, just add to the length of the current block.
            current->addLength(instruction->size());
        }
        // Exit block.
        basicBlocks.constructAndAppend(BytecodeBasicBlock<OpcodeTraits>::ExitBlock, basicBlocks.size());
        basicBlocks.shrinkToFit();
        ASSERT(basicBlocks.last().isExitBlock());
    }
    // After this point, we never change basicBlocks.

    // Link basic blocks together.
    for (unsigned i = 0; i < basicBlocks.size(); i++) {
        auto& block = basicBlocks[i];

        if (block.isEntryBlock() || block.isExitBlock())
            continue;

        bool fallsThrough = true;
        for (unsigned visitedLength = 0; visitedLength < block.totalLength();) {
            auto instruction = instructions.at(block.leaderOffset() + visitedLength);
            auto opcodeID = instruction->opcodeID();

            visitedLength += instruction->size();

            // If we found a terminal bytecode, link to the exit block.
            if (isTerminal(opcodeID)) {
                ASSERT(instruction.offset() + instruction->size() == block.leaderOffset() + block.totalLength());
                linkBlocks(block, basicBlocks.last());
                fallsThrough = false;
                break;
            }

            // If we found a throw, get the HandlerInfo for this instruction to see where we will jump.
            // If there isn't one, treat this throw as a terminal. This is true even if we have a finally
            // block because the finally block will create its own catch, which will generate a HandlerInfo.
            if (isThrow(opcodeID)) {
                ASSERT(instruction.offset() + instruction->size() == block.leaderOffset() + block.totalLength());
                auto* handler = codeBlock->handlerForBytecodeIndex(BytecodeIndex(instruction.offset()));
                fallsThrough = false;
                if (!handler) {
                    linkBlocks(block, basicBlocks.last());
                    break;
                }
                for (auto& otherBlock : basicBlocks) {
                    if (handler->target == otherBlock.leaderOffset()) {
                        linkBlocks(block, otherBlock);
                        break;
                    }
                }
                break;
            }

            // If we found a branch, link to the block(s) that we jump to.
            if (isBranch(opcodeID)) {
                ASSERT(instruction.offset() + instruction->size() == block.leaderOffset() + block.totalLength());
                Vector<typename InstructionStreamType::Offset, 1> bytecodeOffsetsJumpedTo;
                findJumpTargetsForInstruction(codeBlock, instruction, bytecodeOffsetsJumpedTo);

                size_t numberOfJumpTargets = bytecodeOffsetsJumpedTo.size();
                ASSERT(numberOfJumpTargets);
                for (auto& otherBlock : basicBlocks) {
                    if (bytecodeOffsetsJumpedTo.contains(otherBlock.leaderOffset())) {
                        linkBlocks(block, otherBlock);
                        --numberOfJumpTargets;
                        if (!numberOfJumpTargets)
                            break;
                    }
                }
                // numberOfJumpTargets may not be 0 here if there are multiple jumps targeting the same
                // basic blocks (e.g. in a switch type opcode). Since we only decrement numberOfJumpTargets
                // once per basic block, the duplicates are not accounted for. For our purpose here,
                // that doesn't matter because we only need to link to the target block once regardless
                // of how many ways this block can jump there.

                if (isUnconditionalBranch(opcodeID))
                    fallsThrough = false;

                break;
            }
        }

        // If we fall through then link to the next block in program order.
        if (fallsThrough) {
            ASSERT(i + 1 < basicBlocks.size());
            auto& nextBlock = basicBlocks[i + 1];
            linkBlocks(block, nextBlock);
        }
    }

    unsigned index = 0;
    for (auto& basicBlock : basicBlocks) {
        basicBlock.shrinkToFit();
        ASSERT_UNUSED(index, basicBlock.index() == index++);
    }

    return basicBlocks;
}

template<>
auto BytecodeBasicBlock<JSOpcodeTraits>::compute(CodeBlock* codeBlock, const JSInstructionStream& instructions) -> BasicBlockVector
{
    return computeImpl(codeBlock, instructions);
}

template<>
auto BytecodeBasicBlock<JSOpcodeTraits>::compute(UnlinkedCodeBlockGenerator* codeBlock, const JSInstructionStream& instructions) -> BasicBlockVector
{
    return computeImpl(codeBlock, instructions);
}

} // namespace JSC
