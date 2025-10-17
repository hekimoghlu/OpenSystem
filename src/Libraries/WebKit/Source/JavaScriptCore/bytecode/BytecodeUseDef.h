/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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

#include "BytecodeStructs.h"
#include "CodeBlock.h"
#include "Instruction.h"
#include <wtf/Forward.h>

namespace JSC {

void computeUsesForBytecodeIndexImpl(const JSInstruction*, Checkpoint, const ScopedLambda<void(VirtualRegister)>&);
void computeDefsForBytecodeIndexImpl(unsigned, const JSInstruction*, Checkpoint, const ScopedLambda<void(VirtualRegister)>&);

template<typename Block, typename Functor>
void computeUsesForBytecodeIndex(Block* codeBlock, const JSInstruction* instruction, Checkpoint checkpoint, const Functor& functor)
{
    OpcodeID opcodeID = instruction->opcodeID();
    if (opcodeID != op_enter && codeBlock->wasCompiledWithDebuggingOpcodes() && codeBlock->scopeRegister().isValid())
        functor(codeBlock->scopeRegister());

    computeUsesForBytecodeIndexImpl(instruction, checkpoint, scopedLambda<void(VirtualRegister)>(functor));
}

template<typename Block, typename Functor>
void computeDefsForBytecodeIndex(Block* codeBlock, const JSInstruction* instruction, Checkpoint checkpoint, const Functor& functor)
{
    computeDefsForBytecodeIndexImpl(codeBlock->numVars(), instruction, checkpoint, scopedLambda<void(VirtualRegister)>(functor));
}

#undef CALL_FUNCTOR
#undef USES_OR_DEFS
#undef USES
#undef DEFS
} // namespace JSC
