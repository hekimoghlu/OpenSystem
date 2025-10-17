/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

#include "ArrayProfile.h"
#include "BytecodeStructs.h"
#include "Instruction.h"
#include "InterpreterInlines.h"
#include "Opcode.h"

namespace JSC {

enum OpcodeShape {
    AnyOpcodeShape,
    OpCallShape,
};

template<OpcodeShape shape, typename = std::enable_if_t<shape != AnyOpcodeShape>>
inline bool isOpcodeShape(OpcodeID opcodeID)
{
    if (shape == OpCallShape) {
        return opcodeID == op_call
            || opcodeID == op_tail_call
            || opcodeID == op_call_direct_eval
            || opcodeID == op_call_varargs
            || opcodeID == op_call_ignore_result
            || opcodeID == op_tail_call_varargs
            || opcodeID == op_tail_call_forward_arguments
            || opcodeID == op_iterator_open
            || opcodeID == op_iterator_next;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

template<OpcodeShape shape, typename = std::enable_if_t<shape != AnyOpcodeShape>>
inline bool isOpcodeShape(const JSInstruction* instruction)
{
    return isOpcodeShape<shape>(instruction->opcodeID());
}

template<typename T, typename... Args>
void getOpcodeType(OpcodeID opcodeID, Args&&... args)
{

#define CASE(__Op) \
    case __Op::opcodeID: \
        T::template withOpcodeType<__Op>(std::forward<Args>(args)...); \
        break; \

    switch (opcodeID) {
        FOR_EACH_BYTECODE_STRUCT(CASE)
    default:
        ASSERT_NOT_REACHED();
    }

#undef CASE
}

} // namespace JSC
