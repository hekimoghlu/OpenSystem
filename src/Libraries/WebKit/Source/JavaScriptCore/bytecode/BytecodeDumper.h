/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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

#include "BytecodeBasicBlock.h"
#include "BytecodeGeneratorBase.h"
#include "CallLinkInfo.h"
#include "ICStatusMap.h"
#include "Instruction.h"
#include "InstructionStream.h"
#include "StructureStubInfo.h"
#include "WasmOps.h"

namespace JSC {

class BytecodeGraph;

template<typename InstructionStreamType>
class BytecodeDumperBase {
public:
    virtual ~BytecodeDumperBase()
    {
    }

    void printLocationAndOp(typename InstructionStreamType::Offset location, const char* op);

    template<typename T>
    void dumpOperand(const char* operandName, T operand, bool isFirst = false)
    {
        if (!isFirst)
            m_out.print(", ");
        m_out.print(operandName);
        m_out.print(":");
        dumpValue(operand);
    }

    void dumpValue(VirtualRegister);

    template<typename Traits>
    void dumpValue(GenericBoundLabel<Traits>);

    template<typename T>
    void dumpValue(T v) { m_out.print(v); }

protected:
    virtual CString registerName(VirtualRegister) const = 0;
    virtual int outOfLineJumpOffset(typename InstructionStreamType::Offset) const = 0;

    BytecodeDumperBase(PrintStream& out)
        : m_out(out)
    {
    }

    PrintStream& m_out;
    typename InstructionStreamType::Offset m_currentLocation { 0 };
};

template<class Block>
class BytecodeDumper : public BytecodeDumperBase<JSInstructionStream> {
public:
    static void dumpBytecode(Block*, PrintStream& out, const JSInstructionStream::Ref& it, const ICStatusMap& = ICStatusMap());

    BytecodeDumper(Block* block, PrintStream& out)
        : BytecodeDumperBase(out)
        , m_block(block)
    {
    }

    ~BytecodeDumper() override { }

protected:
    Block* block() const { return m_block; }

    void dumpBytecode(const JSInstructionStream::Ref& it, const ICStatusMap&);

    CString registerName(VirtualRegister) const override;
    int outOfLineJumpOffset(JSInstructionStream::Offset) const override;

private:
    virtual CString constantName(VirtualRegister) const;

    Block* m_block;
};

template<class Block>
class CodeBlockBytecodeDumper final : public BytecodeDumper<Block> {
public:
    static void dumpBlock(Block*, const JSInstructionStream&, PrintStream& out, const ICStatusMap& = ICStatusMap());
    static void dumpGraph(Block*, const JSInstructionStream&, BytecodeGraph&, PrintStream& out = WTF::dataFile(), const ICStatusMap& = ICStatusMap());

    void dumpIdentifiers();
    void dumpConstants();
    void dumpExceptionHandlers();
    void dumpSwitchJumpTables();
    void dumpStringSwitchJumpTables();

private:
    using BytecodeDumper<Block>::BytecodeDumper;

    ALWAYS_INLINE VM& vm() const;

    const Identifier& identifier(int index) const;
};

#if ENABLE(WEBASSEMBLY)

namespace Wasm {

class FunctionCodeBlockGenerator;
struct ModuleInformation;

class BytecodeDumper final : public JSC::BytecodeDumperBase<WasmInstructionStream> {
public:
    static void dumpBlock(FunctionCodeBlockGenerator*, const ModuleInformation&, PrintStream& out);

    BytecodeDumper(FunctionCodeBlockGenerator* block, PrintStream& out)
        : BytecodeDumperBase(out)
        , m_block(block)
    {
    }

    ~BytecodeDumper() override { }

    FunctionCodeBlockGenerator* block() const { return m_block; }

    CString registerName(VirtualRegister) const override;
    int outOfLineJumpOffset(WasmInstructionStream::Offset) const override;

private:
    void dumpConstants();
    void dumpExceptionHandlers();
    CString constantName(VirtualRegister index) const;
    CString formatConstant(Type, uint64_t) const;

    FunctionCodeBlockGenerator* m_block;
};

} // namespace Wasm

#endif // ENABLE(WEBASSEMBLY)

}
