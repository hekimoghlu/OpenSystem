/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
#include "ValueProfile.h"

namespace JSC {

template <typename Bytecode>
unsigned valueProfileOffsetFor(const Bytecode& bytecode, unsigned checkpointIndex)
{
    UNUSED_PARAM(checkpointIndex);
    if constexpr (Bytecode::opcodeID == op_iterator_open) {
        switch (checkpointIndex) {
        case OpIteratorOpen::symbolCall: return bytecode.m_iteratorValueProfile;
        case OpIteratorOpen::getNext: return bytecode.m_nextValueProfile;
        default: RELEASE_ASSERT_NOT_REACHED();
        }

    } else if constexpr (Bytecode::opcodeID == op_iterator_next) {
        switch (checkpointIndex) {
        case OpIteratorNext::computeNext: return bytecode.m_nextResultValueProfile;
        case OpIteratorNext::getDone: return bytecode.m_doneValueProfile;
        case OpIteratorNext::getValue: return bytecode.m_valueValueProfile;
        default: RELEASE_ASSERT_NOT_REACHED();
        }
    } else if constexpr (Bytecode::opcodeID == op_instanceof) {
        switch (checkpointIndex) {
        case OpInstanceof::getHasInstance: return bytecode.m_hasInstanceValueProfile;
        case OpInstanceof::getPrototype: return bytecode.m_prototypeValueProfile;
        default: RELEASE_ASSERT_NOT_REACHED();
        }
    } else 
        return bytecode.m_valueProfile;
}

template<typename Bytecode>
Operand destinationFor(const Bytecode& bytecode, unsigned checkpointIndex, JITType type = JITType::None)
{
    UNUSED_PARAM(checkpointIndex);
    if constexpr (Bytecode::opcodeID == op_iterator_open) {
        switch (checkpointIndex) {
        case OpIteratorOpen::symbolCall: return bytecode.m_iterator;
        case OpIteratorOpen::getNext: return bytecode.m_next;
        default: RELEASE_ASSERT_NOT_REACHED();
        }
        return { };
    } else if constexpr (Bytecode::opcodeID == op_iterator_next) {
        switch (checkpointIndex) {
        case OpIteratorNext::computeNext: {
            if (type == JITType::DFGJIT || type == JITType::FTLJIT)
                return Operand::tmp(OpIteratorNext::nextResult);
            return bytecode.m_value; // We reuse value as a temp because its either not used in subsequent bytecodes or written as the temp object .
        }
        case OpIteratorNext::getDone: return bytecode.m_done;
        case OpIteratorNext::getValue: return bytecode.m_value;
        default: RELEASE_ASSERT_NOT_REACHED();
        }
        return { };
    } else if constexpr (Bytecode::opcodeID == op_call_ignore_result) {
        return { };
    } else
        return bytecode.m_dst;
}

// Call-like opcode locations

template<typename Bytecode>
VirtualRegister calleeFor(const Bytecode& bytecode, unsigned checkpointIndex)
{
    UNUSED_PARAM(checkpointIndex);
    if constexpr (Bytecode::opcodeID == op_iterator_open) {
        ASSERT(checkpointIndex == OpIteratorOpen::symbolCall);
        return bytecode.m_symbolIterator;
    } else if constexpr (Bytecode::opcodeID == op_iterator_next) {
        ASSERT(checkpointIndex == OpIteratorNext::computeNext);
        return bytecode.m_next;
    } else
        return bytecode.m_callee;
}

template<typename Bytecode>
unsigned argumentCountIncludingThisFor(const Bytecode& bytecode, unsigned checkpointIndex)
{
    UNUSED_PARAM(checkpointIndex);
    if constexpr (Bytecode::opcodeID == op_iterator_open) {
        ASSERT(checkpointIndex == OpIteratorOpen::symbolCall);
        return 1;
    } else if constexpr (Bytecode::opcodeID == op_iterator_next) {
        ASSERT(checkpointIndex == OpIteratorNext::computeNext);
        return 1;
    } else
        return bytecode.m_argc;
}

template<typename Bytecode>
ptrdiff_t stackOffsetInRegistersForCall(const Bytecode& bytecode, unsigned checkpointIndex)
{
    UNUSED_PARAM(checkpointIndex);
    if constexpr (Bytecode::opcodeID == op_iterator_open) {
        ASSERT(checkpointIndex == OpIteratorOpen::symbolCall);
        return CallFrame::headerSizeInRegisters - bytecode.m_iterable.offset();
    } else if constexpr (Bytecode::opcodeID == op_iterator_next) {
        ASSERT(checkpointIndex == OpIteratorNext::computeNext);
        return CallFrame::headerSizeInRegisters - bytecode.m_iterator.offset();
    } else
        return bytecode.m_argv;
}

template<typename BytecodeMetadata>
CallLinkInfo& callLinkInfoFor(BytecodeMetadata& metadata, unsigned checkpointIndex)
{
    UNUSED_PARAM(checkpointIndex);
    return metadata.m_callLinkInfo;
}

} // namespace JSC
