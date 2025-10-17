/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
#include "PreciseJumpTargets.h"

#include "JSCJSValueInlines.h"
#include "PreciseJumpTargetsInlines.h"

namespace JSC {

template <size_t vectorSize, typename Block>
static void getJumpTargetsForInstruction(Block* codeBlock, const JSInstructionStream::Ref& instruction, Vector<JSInstructionStream::Offset, vectorSize>& out)
{
    extractStoredJumpTargetsForInstruction(codeBlock, instruction, [&](int32_t relativeOffset) {
        out.append(instruction.offset() + relativeOffset);
    });
    OpcodeID opcodeID = instruction->opcodeID();
    // op_loop_hint does not have jump target stored in bytecode instructions.
    if (opcodeID == op_loop_hint)
        out.append(instruction.offset());
}

enum class ComputePreciseJumpTargetsMode {
    FollowCodeBlockClaim,
    ForceCompute,
};

template<ComputePreciseJumpTargetsMode Mode, typename Block, size_t vectorSize>
void computePreciseJumpTargetsInternal(Block* codeBlock, const JSInstructionStream& instructions, Vector<JSInstructionStream::Offset, vectorSize>& out)
{
    ASSERT(out.isEmpty());

    // The code block has a superset of the jump targets. So if it claims to have none, we are done.
    if (Mode == ComputePreciseJumpTargetsMode::FollowCodeBlockClaim && !codeBlock->numberOfJumpTargets())
        return;
    
    for (unsigned i = codeBlock->numberOfExceptionHandlers(); i--;) {
        out.append(codeBlock->exceptionHandler(i).target);
        out.append(codeBlock->exceptionHandler(i).start);
        out.append(codeBlock->exceptionHandler(i).end);
    }

    for (const auto& instruction : instructions) {
        getJumpTargetsForInstruction(codeBlock, instruction, out);
    }
    
    std::sort(out.begin(), out.end());
    
    // We will have duplicates, and we must remove them.
    unsigned toIndex = 0;
    unsigned fromIndex = 0;
    unsigned lastValue = UINT_MAX;
    while (fromIndex < out.size()) {
        unsigned value = out[fromIndex++];
        if (value == lastValue)
            continue;
        out[toIndex++] = value;
        lastValue = value;
    }
    out.shrinkCapacity(toIndex);
}

void computePreciseJumpTargets(CodeBlock* codeBlock, Vector<JSInstructionStream::Offset, 32>& out)
{
    computePreciseJumpTargetsInternal<ComputePreciseJumpTargetsMode::FollowCodeBlockClaim>(codeBlock, codeBlock->instructions(), out);
}

void computePreciseJumpTargets(CodeBlock* codeBlock, const JSInstructionStream& instructions, Vector<JSInstructionStream::Offset, 32>& out)
{
    computePreciseJumpTargetsInternal<ComputePreciseJumpTargetsMode::FollowCodeBlockClaim>(codeBlock, instructions, out);
}

void computePreciseJumpTargets(UnlinkedCodeBlockGenerator* codeBlock, const JSInstructionStream& instructions, Vector<JSInstructionStream::Offset, 32>& out)
{
    computePreciseJumpTargetsInternal<ComputePreciseJumpTargetsMode::FollowCodeBlockClaim>(codeBlock, instructions, out);
}

void recomputePreciseJumpTargets(UnlinkedCodeBlockGenerator* codeBlock, const JSInstructionStream& instructions, Vector<JSInstructionStream::Offset>& out)
{
    computePreciseJumpTargetsInternal<ComputePreciseJumpTargetsMode::ForceCompute>(codeBlock, instructions, out);
}

void findJumpTargetsForInstruction(CodeBlock* codeBlock, const JSInstructionStream::Ref& instruction, Vector<JSInstructionStream::Offset, 1>& out)
{
    getJumpTargetsForInstruction(codeBlock, instruction, out);
}

void findJumpTargetsForInstruction(UnlinkedCodeBlockGenerator* codeBlock, const JSInstructionStream::Ref& instruction, Vector<JSInstructionStream::Offset, 1>& out)
{
    getJumpTargetsForInstruction(codeBlock, instruction, out);
}

} // namespace JSC

