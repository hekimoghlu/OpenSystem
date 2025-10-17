/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#include "WasmFunctionCodeBlockGenerator.h"

#if ENABLE(WEBASSEMBLY)

#include "InstructionStream.h"
#include "VirtualRegister.h"
#include <wtf/FixedVector.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace Wasm {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FunctionCodeBlockGenerator);

void FunctionCodeBlockGenerator::setInstructions(std::unique_ptr<WasmInstructionStream> instructions)
{
    m_instructions = WTFMove(instructions);
    m_instructionsRawPointer = m_instructions->rawPointer();
}

void FunctionCodeBlockGenerator::addOutOfLineJumpTarget(WasmInstructionStream::Offset bytecodeOffset, int target)
{
    RELEASE_ASSERT(target);
    m_outOfLineJumpTargets.set(bytecodeOffset, target);
}

WasmInstructionStream::Offset FunctionCodeBlockGenerator::outOfLineJumpOffset(WasmInstructionStream::Offset bytecodeOffset)
{
    ASSERT(m_outOfLineJumpTargets.contains(bytecodeOffset));
    return m_outOfLineJumpTargets.get(bytecodeOffset);
}

unsigned FunctionCodeBlockGenerator::addSignature(const TypeDefinition& signature)
{
    unsigned index = m_signatures.size();
    m_signatures.append(&signature);
    return index;
}

auto FunctionCodeBlockGenerator::addJumpTable(size_t numberOfEntries) -> JumpTable&
{
    m_jumpTables.append(JumpTable(numberOfEntries));
    return m_jumpTables.last();
}

unsigned FunctionCodeBlockGenerator::numberOfJumpTables() const
{
    return m_jumpTables.size();
}

void FunctionCodeBlockGenerator::setTailCall(uint32_t functionIndex, bool isImportedFunctionFromFunctionIndexSpace)
{
    m_tailCallSuccessors.set(functionIndex);
    if (isImportedFunctionFromFunctionIndexSpace)
        setTailCallClobbersInstance();
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
