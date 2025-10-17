/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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

#if ENABLE(WEBASSEMBLY)

#include "BytecodeConventions.h"
#include "HandlerInfo.h"
#include "InstructionStream.h"
#include "MacroAssemblerCodeRef.h"
#include "WasmFormat.h"
#include "WasmHandlerInfo.h"
#include "WasmLLIntTierUpCounter.h"
#include "WasmOps.h"
#include <wtf/FixedBitVector.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC {

class JITCode;
class LLIntOffsetsExtractor;

template<typename Traits>
class BytecodeGeneratorBase;

namespace Wasm {

class LLIntCallee;
class TypeDefinition;
struct GeneratorTraits;

struct JumpTableEntry {
    int target { 0 };
    unsigned startOffset;
    unsigned dropCount;
    unsigned keepCount;
};

using JumpTable = FixedVector<JumpTableEntry>;

class FunctionCodeBlockGenerator {
    WTF_MAKE_TZONE_ALLOCATED(FunctionCodeBlockGenerator);
    WTF_MAKE_NONCOPYABLE(FunctionCodeBlockGenerator);

    friend BytecodeGeneratorBase<GeneratorTraits>;
    friend LLIntOffsetsExtractor;
    friend class LLIntGenerator;
    friend class LLIntCallee;

public:
    FunctionCodeBlockGenerator(FunctionCodeIndex functionIndex)
        : m_functionIndex(functionIndex)
    {
    }

    FunctionCodeIndex functionIndex() const { return m_functionIndex; }
    unsigned numVars() const { return m_numVars; }
    unsigned numCalleeLocals() const { return m_numCalleeLocals; }
    uint32_t numArguments() const { return m_numArguments; }
    const Vector<Type>& constantTypes() const { return m_constantTypes; }
    const Vector<uint64_t>& constants() const { return m_constants; }
    const Vector<uint64_t>& constantRegisters() const { return m_constants; }
    const WasmInstructionStream& instructions() const { return *m_instructions; }
    const BitVector& tailCallSuccessors() const { return m_tailCallSuccessors; }
    bool tailCallClobbersInstance() const { return m_tailCallClobbersInstance ; }
    void setTailCall(uint32_t, bool);
    void setTailCallClobbersInstance() { m_tailCallClobbersInstance = true; }

    void setNumVars(unsigned numVars) { m_numVars = numVars; }
    void setNumCalleeLocals(unsigned numCalleeLocals) { m_numCalleeLocals = numCalleeLocals; }

    ALWAYS_INLINE uint64_t getConstant(VirtualRegister reg) const { return m_constants[reg.toConstantIndex()]; }
    ALWAYS_INLINE Type getConstantType(VirtualRegister reg) const
    {
        ASSERT(Options::dumpGeneratedWasmBytecodes());
        return m_constantTypes[reg.toConstantIndex()];
    }

    void setInstructions(std::unique_ptr<WasmInstructionStream>);
    void addJumpTarget(WasmInstructionStream::Offset jumpTarget) { m_jumpTargets.append(jumpTarget); }
    WasmInstructionStream::Offset numberOfJumpTargets() { return m_jumpTargets.size(); }
    WasmInstructionStream::Offset lastJumpTarget() { return m_jumpTargets.last(); }

    void addOutOfLineJumpTarget(WasmInstructionStream::Offset, int target);
    WasmInstructionStream::Offset outOfLineJumpOffset(WasmInstructionStream::Offset);
    WasmInstructionStream::Offset outOfLineJumpOffset(const WasmInstructionStream::Ref& instruction)
    {
        return outOfLineJumpOffset(instruction.offset());
    }

    inline WasmInstructionStream::Offset bytecodeOffset(const WasmInstruction* returnAddress)
    {
        const auto* instructionsBegin = m_instructions->at(0).ptr();
        const auto* instructionsEnd = reinterpret_cast<const WasmInstruction*>(reinterpret_cast<uintptr_t>(instructionsBegin) + m_instructions->size());
        RELEASE_ASSERT(returnAddress >= instructionsBegin && returnAddress < instructionsEnd);
        return returnAddress - instructionsBegin;
    }

    UncheckedKeyHashMap<WasmInstructionStream::Offset, LLIntTierUpCounter::OSREntryData>& tierUpCounter() { return m_tierUpCounter; }

    unsigned addSignature(const TypeDefinition&);

    JumpTable& addJumpTable(size_t numberOfEntries);
    unsigned numberOfJumpTables() const;

    size_t numberOfExceptionHandlers() const { return m_exceptionHandlers.size(); }
    UnlinkedHandlerInfo& exceptionHandler(int index) { return m_exceptionHandlers[index]; }
    void addExceptionHandler(const UnlinkedHandlerInfo& handler) { m_exceptionHandlers.append(handler); }

private:
    using OutOfLineJumpTargets = UncheckedKeyHashMap<WasmInstructionStream::Offset, int>;

    FunctionCodeIndex m_functionIndex;

    // Used for the number of WebAssembly locals, as in https://webassembly.github.io/spec/core/syntax/modules.html#syntax-local
    unsigned m_numVars { 0 };
    // Number of VirtualRegister. The naming is unfortunate, but has to match UnlinkedCodeBlock
    unsigned m_numCalleeLocals { 0 };
    uint32_t m_numArguments { 0 };
    bool m_tailCallClobbersInstance { false };
    Vector<Type> m_constantTypes;
    Vector<uint64_t> m_constants;
    std::unique_ptr<WasmInstructionStream> m_instructions;
    const void* m_instructionsRawPointer { nullptr };
    Vector<WasmInstructionStream::Offset> m_jumpTargets;
    Vector<const TypeDefinition*> m_signatures;
    OutOfLineJumpTargets m_outOfLineJumpTargets;
    UncheckedKeyHashMap<WasmInstructionStream::Offset, LLIntTierUpCounter::OSREntryData> m_tierUpCounter;
    Vector<JumpTable> m_jumpTables;
    Vector<UnlinkedHandlerInfo> m_exceptionHandlers;
    BitVector m_tailCallSuccessors;
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
