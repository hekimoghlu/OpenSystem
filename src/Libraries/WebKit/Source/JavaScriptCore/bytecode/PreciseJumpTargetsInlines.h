/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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

#include "BytecodeGenerator.h"
#include "BytecodeStructs.h"
#include "InterpreterInlines.h"
#include "Opcode.h"
#include "PreciseJumpTargets.h"

namespace JSC {

#define SWITCH_JMP(CASE_OP, SWITCH_CASE) \
    switch (instruction->opcodeID()) { \
    CASE_OP(OpJmp) \
    \
    CASE_OP(OpJtrue) \
    CASE_OP(OpJfalse) \
    CASE_OP(OpJeqNull) \
    CASE_OP(OpJneqNull) \
    CASE_OP(OpJundefinedOrNull) \
    CASE_OP(OpJnundefinedOrNull) \
    CASE_OP(OpJeqPtr) \
    CASE_OP(OpJneqPtr) \
    \
    CASE_OP(OpJless) \
    CASE_OP(OpJlesseq) \
    CASE_OP(OpJgreater) \
    CASE_OP(OpJgreatereq) \
    CASE_OP(OpJnless) \
    CASE_OP(OpJnlesseq) \
    CASE_OP(OpJngreater) \
    CASE_OP(OpJngreatereq) \
    CASE_OP(OpJeq) \
    CASE_OP(OpJneq) \
    CASE_OP(OpJstricteq) \
    CASE_OP(OpJnstricteq) \
    CASE_OP(OpJbelow) \
    CASE_OP(OpJbeloweq) \
    case op_switch_imm: { \
        auto bytecode = instruction->as<OpSwitchImm>(); \
        auto& table = codeBlock->unlinkedSwitchJumpTable(bytecode.m_tableIndex); \
        for (unsigned i = table.m_branchOffsets.size(); i--;) \
            SWITCH_CASE(table.m_branchOffsets[i]); \
        SWITCH_CASE(table.m_defaultOffset); \
        break; \
    } \
    case op_switch_char: { \
        auto bytecode = instruction->as<OpSwitchChar>(); \
        auto& table = codeBlock->unlinkedSwitchJumpTable(bytecode.m_tableIndex); \
        for (unsigned i = table.m_branchOffsets.size(); i--;) \
            SWITCH_CASE(table.m_branchOffsets[i]); \
        SWITCH_CASE(table.m_defaultOffset); \
        break; \
    } \
    case op_switch_string: { \
        auto bytecode = instruction->as<OpSwitchString>(); \
        auto& table = codeBlock->unlinkedStringSwitchJumpTable(bytecode.m_tableIndex); \
        for (auto& entry : table.m_offsetTable) \
            SWITCH_CASE(entry.value.m_branchOffset); \
        SWITCH_CASE(table.m_defaultOffset); \
        break; \
    } \
    default: \
        break; \
    } \


template<typename Block>
inline int jumpTargetForInstruction(Block* codeBlock, const JSInstructionStream::Ref& instruction, unsigned target)
{
    if (target)
        return target;
    return codeBlock->outOfLineJumpOffset(instruction);
}

template<typename UncheckedKeyHashMap>
inline int jumpTargetForInstruction(UncheckedKeyHashMap& outOfLineJumpTargets, const JSInstructionStream::Ref& instruction, unsigned target)
{
    if (target)
        return target;
    ASSERT(outOfLineJumpTargets.contains(instruction.offset()));
    return outOfLineJumpTargets.get(instruction.offset());
}

template<typename Op, typename Block>
inline int jumpTargetForInstruction(Block&& codeBlock, const JSInstructionStream::Ref& instruction)
{
    auto bytecode = instruction->as<Op>();
    return jumpTargetForInstruction(codeBlock, instruction, bytecode.m_targetLabel);
}

template<typename Block, typename Function>
inline void extractStoredJumpTargetsForInstruction(Block&& codeBlock, const JSInstructionStream::Ref& instruction, const Function& function)
{
#define CASE_OP(__op) \
    case __op::opcodeID: \
        function(jumpTargetForInstruction<__op>(codeBlock, instruction)); \
        break;

#define SWITCH_CASE(__target) \
    function(__target)

SWITCH_JMP(CASE_OP, SWITCH_CASE)

#undef CASE_OP
#undef SWITCH_CASE
}

template<typename Block, typename Function, typename CodeBlockOrHashMap>
inline void updateStoredJumpTargetsForInstruction(Block&& codeBlock, unsigned finalOffset, JSInstructionStream::MutableRef instruction, const Function& function, CodeBlockOrHashMap& codeBlockOrHashMap)
{
#define CASE_OP(__op) \
    case __op::opcodeID: { \
        int32_t target = jumpTargetForInstruction<__op>(codeBlockOrHashMap, instruction); \
        int32_t newTarget = function(target); \
        instruction->cast<__op>()->setTargetLabel(BoundLabel(newTarget), [&]() { \
            codeBlock->addOutOfLineJumpTarget(finalOffset + instruction.offset(), newTarget); \
            return BoundLabel(); \
        }); \
        break; \
    }

#define SWITCH_CASE(__target) \
    do { \
        int32_t target = __target; \
        __target = function(target); \
    } while (false)

SWITCH_JMP(CASE_OP, SWITCH_CASE)

#undef CASE_OP
#undef JMP_TARGET
}

template<typename Block, typename Function>
inline void updateStoredJumpTargetsForInstruction(Block* codeBlock, unsigned finalOffset, JSInstructionStream::MutableRef instruction, Function function)
{
    updateStoredJumpTargetsForInstruction(codeBlock, finalOffset, instruction, function, codeBlock);
}

} // namespace JSC
