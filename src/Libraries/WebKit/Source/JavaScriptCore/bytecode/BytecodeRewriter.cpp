/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#include "BytecodeRewriter.h"

#include "JSCJSValueInlines.h"
#include "PreciseJumpTargetsInlines.h"
#include <wtf/BubbleSort.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

void BytecodeRewriter::applyModification()
{
    for (size_t insertionIndex = m_insertions.size(); insertionIndex--;) {
        Insertion& insertion = m_insertions[insertionIndex];
        if (insertion.type == Insertion::Type::Remove)
            m_writer.m_instructions.remove(insertion.index.bytecodeOffset, insertion.length());
        else {
            if (insertion.includeBranch == IncludeBranch::Yes) {
                int finalOffset = insertion.index.bytecodeOffset + calculateDifference(m_insertions.begin(), m_insertions.begin() + insertionIndex);
                adjustJumpTargetsInFragment(finalOffset, insertion);
            }
            m_writer.m_instructions.insertVector(insertion.index.bytecodeOffset, insertion.instructions.m_instructions);
        }
    }
    m_insertions.clear();
}

void BytecodeRewriter::execute()
{
    WTF::bubbleSort(m_insertions.begin(), m_insertions.end(), [] (const Insertion& lhs, const Insertion& rhs) {
        return lhs.index < rhs.index;
    });

    m_codeBlock->applyModification(*this, m_writer);
}

void BytecodeRewriter::adjustJumpTargetsInFragment(unsigned finalOffset, Insertion& insertion)
{
    for (auto& instruction : insertion.instructions) {
        if (isBranch(instruction->opcodeID())) {
            unsigned bytecodeOffset = finalOffset + instruction.offset();
            updateStoredJumpTargetsForInstruction(m_codeBlock, finalOffset, instruction, [&](int32_t label) {
                int absoluteOffset = adjustAbsoluteOffset(label);
                return absoluteOffset - static_cast<int>(bytecodeOffset);
            });
        }
    }
}

void BytecodeRewriter::insertImpl(InsertionPoint insertionPoint, IncludeBranch includeBranch, JSInstructionStreamWriter&& writer)
{
    ASSERT(insertionPoint.position == Position::Before || insertionPoint.position == Position::After);
    m_insertions.append(Insertion {
        insertionPoint,
        Insertion::Type::Insert,
        includeBranch,
        0,
        WTFMove(writer)
    });
}

int32_t BytecodeRewriter::adjustJumpTarget(InsertionPoint startPoint, InsertionPoint jumpTargetPoint)
{
    if (startPoint < jumpTargetPoint) {
        int jumpTarget = jumpTargetPoint.bytecodeOffset;
        auto start = std::lower_bound(m_insertions.begin(), m_insertions.end(), startPoint, [&] (const Insertion& insertion, InsertionPoint startPoint) {
            return insertion.index < startPoint;
        });
        if (start != m_insertions.end()) {
            auto end = std::lower_bound(m_insertions.begin(), m_insertions.end(), jumpTargetPoint, [&] (const Insertion& insertion, InsertionPoint jumpTargetPoint) {
                return insertion.index < jumpTargetPoint;
            });
            jumpTarget += calculateDifference(start, end);
        }
        return jumpTarget - startPoint.bytecodeOffset;
    }

    if (startPoint == jumpTargetPoint)
        return 0;

    return -adjustJumpTarget(jumpTargetPoint, startPoint);
}

// FIXME: unit test the logic in this method
// https://bugs.webkit.org/show_bug.cgi?id=190950
void BytecodeRewriter::adjustJumpTargets()
{
    auto currentInsertion = m_insertions.begin();
    auto outOfLineJumpTargets = m_codeBlock->replaceOutOfLineJumpTargets();

    int offset = 0;
    for (JSInstructionStream::Offset i = 0; i < m_writer.size();) {
        int before = 0;
        int after = 0;
        int remove = 0;
        while (currentInsertion != m_insertions.end() && static_cast<JSInstructionStream::Offset>(currentInsertion->index.bytecodeOffset) == i) {
            auto size = currentInsertion->length();
            if (currentInsertion->type == Insertion::Type::Remove)
                remove += size;
            else if (currentInsertion->index.position == Position::Before)
                before += size;
            else if (currentInsertion->index.position == Position::After)
                after += size;
            ++currentInsertion;
        }

        offset += before;

        if (!remove) {
            auto instruction = m_writer.ref(i);
            updateStoredJumpTargetsForInstruction(m_codeBlock, offset, instruction, [&](int32_t relativeOffset) {
                return adjustJumpTarget(instruction.offset(), instruction.offset() + relativeOffset);
            }, outOfLineJumpTargets);
            i += instruction->size();
        } else {
            offset -= remove;
            i += remove;
        }

        offset += after;
    }
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
