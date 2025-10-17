/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
#include "BytecodeDumper.h"

#include "BytecodeGenerator.h"
#include "BytecodeGraph.h"
#include "BytecodeStructs.h"
#include "CodeBlock.h"
#include "JSCJSValueInlines.h"
#include "UnlinkedCodeBlockGenerator.h"
#include "UnlinkedMetadataTableInlines.h"
#include "WasmFunctionCodeBlockGenerator.h"
#include "WasmGeneratorTraits.h"
#include "WasmModuleInformation.h"
#include "WasmTypeDefinitionInlines.h"
#include <wtf/text/MakeString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

template<typename InstructionStreamType>
void BytecodeDumperBase<InstructionStreamType>::printLocationAndOp(typename InstructionStreamType::Offset location, const char* op)
{
    m_currentLocation = location;
    m_out.printf("[%4u] %-18s ", location, op);
}

template<typename InstructionStreamType>
void BytecodeDumperBase<InstructionStreamType>::dumpValue(VirtualRegister reg)
{
    m_out.printf("%s", registerName(reg).data());
}

template<typename InstructionStreamType>
template<typename Traits>
void BytecodeDumperBase<InstructionStreamType>::dumpValue(GenericBoundLabel<Traits> label)
{
    int target = label.target();
    if (!target)
        target = outOfLineJumpOffset(m_currentLocation);
    auto targetOffset = target + m_currentLocation;
    m_out.print(target, "(->", targetOffset, ")");
}

template void BytecodeDumperBase<JSInstructionStream>::dumpValue(GenericBoundLabel<JSGeneratorTraits>);

#if ENABLE(WEBASSEMBLY)
template void BytecodeDumperBase<WasmInstructionStream>::dumpValue(GenericBoundLabel<Wasm::GeneratorTraits>);
#endif // ENABLE(WEBASSEMBLY)

template<class Block>
CString BytecodeDumper<Block>::registerName(VirtualRegister r) const
{
    if (r.isConstant())
        return constantName(r);

    return toCString(r);
}

template <class Block>
int BytecodeDumper<Block>::outOfLineJumpOffset(JSInstructionStream::Offset offset) const
{
    return m_block->outOfLineJumpOffset(offset);
}

template<class Block>
CString BytecodeDumper<Block>::constantName(VirtualRegister reg) const
{
    if (reg.toConstantIndex() >= (int) block()->constantRegisters().size())
        return toCString("INVALID_CONSTANT(", reg, ")");
    auto value = block()->getConstant(reg);
    return toCString(value, "(", reg, ")");
}

template<class Block>
void BytecodeDumper<Block>::dumpBytecode(const JSInstructionStream::Ref& it, const ICStatusMap&)
{
    ::JSC::dumpBytecode(this, it.offset(), it.ptr());
    this->m_out.print("\n");
}

template<class Block>
void BytecodeDumper<Block>::dumpBytecode(Block* block, PrintStream& out, const JSInstructionStream::Ref& it, const ICStatusMap& statusMap)
{
    BytecodeDumper dumper(block, out);
    dumper.dumpBytecode(it, statusMap);
}

template<class Block>
VM& CodeBlockBytecodeDumper<Block>::vm() const
{
    return this->block()->vm();
}

template<class Block>
const Identifier& CodeBlockBytecodeDumper<Block>::identifier(int index) const
{
    return this->block()->identifier(index);
}

template<class Block>
void CodeBlockBytecodeDumper<Block>::dumpIdentifiers()
{
    if (size_t count = this->block()->numberOfIdentifiers()) {
        this->m_out.printf("\nIdentifiers:\n");
        size_t i = 0;
        do {
            this->m_out.print("  id", static_cast<unsigned>(i), " = ", identifier(i), "\n");
            ++i;
        } while (i != count);
    }
}

template<class Block>
void CodeBlockBytecodeDumper<Block>::dumpConstants()
{
    if (!this->block()->constantRegisters().isEmpty()) {
        this->m_out.printf("\nConstants:\n");
        size_t i = 0;
        for (const auto& constant : this->block()->constantRegisters()) {
            const char* sourceCodeRepresentationDescription = nullptr;
            switch (this->block()->constantSourceCodeRepresentation(i)) {
            case SourceCodeRepresentation::Double:
                sourceCodeRepresentationDescription = ": in source as double";
                break;
            case SourceCodeRepresentation::Integer:
                sourceCodeRepresentationDescription = ": in source as integer";
                break;
            case SourceCodeRepresentation::Other:
                sourceCodeRepresentationDescription = "";
                break;
            case SourceCodeRepresentation::LinkTimeConstant:
                sourceCodeRepresentationDescription = ": in source as link-time-constant";
                break;
            }
            this->m_out.printf("   k%u = %s%s\n", static_cast<unsigned>(i), toCString(constant.get()).data(), sourceCodeRepresentationDescription);
            ++i;
        }
    }
}

template<class Block>
void CodeBlockBytecodeDumper<Block>::dumpExceptionHandlers()
{
    if (unsigned count = this->block()->numberOfExceptionHandlers()) {
        this->m_out.printf("\nException Handlers:\n");
        unsigned i = 0;
        do {
            const auto& handler = this->block()->exceptionHandler(i);
            this->m_out.printf("\t %d: { start: [%4d] end: [%4d] target: [%4d] } %s\n", i + 1, handler.start, handler.end, handler.target, handler.typeName());
            ++i;
        } while (i < count);
    }
}

template<class Block>
void CodeBlockBytecodeDumper<Block>::dumpSwitchJumpTables()
{
    if (unsigned count = this->block()->numberOfUnlinkedSwitchJumpTables()) {
        this->m_out.printf("Switch Jump Tables:\n");
        unsigned i = 0;
        do {
            this->m_out.printf("  %1d = {\n", i);
            const auto& unlinkedTable = this->block()->unlinkedSwitchJumpTable(i);
            int entry = 0;
            auto end = unlinkedTable.m_branchOffsets.end();
            for (auto iter = unlinkedTable.m_branchOffsets.begin(); iter != end; ++iter, ++entry) {
                if (!*iter)
                    continue;
                this->m_out.printf("\t\t%4d => %04d\n", entry + unlinkedTable.m_min, *iter);
            }
            this->m_out.printf("\t\tdefault => %04d\n", unlinkedTable.m_defaultOffset);
            this->m_out.printf("      }\n");
            ++i;
        } while (i < count);
    }
}

template<class Block>
void CodeBlockBytecodeDumper<Block>::dumpStringSwitchJumpTables()
{
    if (unsigned count = this->block()->numberOfUnlinkedStringSwitchJumpTables()) {
        this->m_out.printf("\nString Switch Jump Tables:\n");
        unsigned i = 0;
        do {
            this->m_out.printf("  %1d = {\n", i);
            auto& unlinkedTable = this->block()->unlinkedStringSwitchJumpTable(i);
            for (const auto& entry : unlinkedTable.m_offsetTable)
                this->m_out.printf("\t\t\"%s\" => %04d\n", entry.key->utf8().data(), entry.value.m_branchOffset);
            this->m_out.printf("\t\tdefault => %04d\n", unlinkedTable.m_defaultOffset);
            this->m_out.printf("      }\n");
            ++i;
        } while (i < count);
    }
}

template <typename Block>
static void dumpHeader(Block* block, const JSInstructionStream& instructions, PrintStream& out)
{
    size_t instructionCount = 0;
    size_t wide16InstructionCount = 0;
    size_t wide32InstructionCount = 0;
    size_t instructionWithMetadataCount = 0;

    for (const auto& instruction : instructions) {
        if (instruction->isWide16())
            ++wide16InstructionCount;
        else if (instruction->isWide32())
            ++wide32InstructionCount;
        if (instruction->hasMetadata())
            ++instructionWithMetadataCount;
        ++instructionCount;
    }

    out.print(*block);
    out.printf(
        ": %lu instructions (%lu 16-bit instructions, %lu 32-bit instructions, %lu instructions with metadata); %lu bytes (%lu metadata bytes); %d parameter(s); %d callee register(s); %d variable(s)",
        static_cast<unsigned long>(instructionCount),
        static_cast<unsigned long>(wide16InstructionCount),
        static_cast<unsigned long>(wide32InstructionCount),
        static_cast<unsigned long>(instructionWithMetadataCount),
        static_cast<unsigned long>(instructions.sizeInBytes() + block->metadataSizeInBytes()),
        static_cast<unsigned long>(block->metadataSizeInBytes()),
        block->numParameters(), block->numCalleeLocals(), block->numVars());
    out.print("; scope at ", block->scopeRegister());
    out.printf("\n");
}

template <typename Dumper>
static void dumpFooter(Dumper& dumper)
{
    dumper.dumpIdentifiers();
    dumper.dumpConstants();
    dumper.dumpExceptionHandlers();
    dumper.dumpSwitchJumpTables();
    dumper.dumpStringSwitchJumpTables();
}

template<class Block>
void CodeBlockBytecodeDumper<Block>::dumpBlock(Block* block, const JSInstructionStream& instructions, PrintStream& out, const ICStatusMap& statusMap)
{
    dumpHeader(block, instructions, out);

    CodeBlockBytecodeDumper<Block> dumper(block, out);
    for (const auto& it : instructions)
        dumper.dumpBytecode(it, statusMap);

    dumpFooter(dumper);

    out.printf("\n");
}

template<class Block>
void CodeBlockBytecodeDumper<Block>::dumpGraph(Block* block, const JSInstructionStream& instructions, BytecodeGraph& graph, PrintStream& out, const ICStatusMap& icStatusMap)
{
    dumpHeader(block, instructions, out);

    CodeBlockBytecodeDumper<Block> dumper(block, out);

    out.printf("\n");

    Vector<Vector<unsigned>> predecessors(graph.size());
    for (auto& block : graph) {
        if (block.isEntryBlock() || block.isExitBlock())
            continue;
        for (auto successorIndex : block.successors()) {
            if (!predecessors[successorIndex].contains(block.index()))
                predecessors[successorIndex].append(block.index());
        }
    }

    for (auto& block : graph) {
        if (block.isEntryBlock() || block.isExitBlock())
            continue;

        out.print("bb#", block.index(), "\n");

        out.print("Predecessors: [");
        for (unsigned predecessor : predecessors[block.index()]) {
            if (!graph[predecessor].isEntryBlock())
                out.print(" #", predecessor);
        }
        out.print(" ]\n");

        for (unsigned i = 0; i < block.totalLength(); ) {
            auto& currentInstruction = instructions.at(i + block.leaderOffset());
            dumper.dumpBytecode(currentInstruction, icStatusMap);
            i += currentInstruction.ptr()->size();
        }

        out.print("Successors: [");
        for (unsigned successor : block.successors()) {
            if (!graph[successor].isExitBlock())
                out.print(" #", successor);
        }
        out.print(" ]\n\n");
    }

    dumpFooter(dumper);

    out.printf("\n");
}

template class BytecodeDumperBase<JSInstructionStream>;
template class BytecodeDumper<CodeBlock>;
template class CodeBlockBytecodeDumper<UnlinkedCodeBlockGenerator>;
template class CodeBlockBytecodeDumper<CodeBlock>;

#if ENABLE(WEBASSEMBLY)

template class BytecodeDumperBase<WasmInstructionStream>;

namespace Wasm {

void BytecodeDumper::dumpBlock(FunctionCodeBlockGenerator* block, const ModuleInformation& moduleInformation, PrintStream& out)
{
    size_t instructionCount = 0;
    size_t wide16InstructionCount = 0;
    size_t wide32InstructionCount = 0;

    for (auto it = block->instructions().begin(); it != block->instructions().end(); it += it->size()) {
        if (it->isWide16())
            ++wide16InstructionCount;
        else if (it->isWide32())
            ++wide32InstructionCount;
        ++instructionCount;
    }

    size_t functionIndexSpace = moduleInformation.importFunctionCount() + block->functionIndex();
    out.print(makeString(IndexOrName(functionIndexSpace, moduleInformation.nameSection->get(functionIndexSpace))));

    const auto& function = moduleInformation.functions[block->functionIndex()];
    TypeIndex typeIndex = moduleInformation.internalFunctionTypeIndices[block->functionIndex()];
    const auto& typeDefinition = TypeInformation::get(typeIndex);
    out.print(" : ", typeDefinition, "\n");
    out.print("wasm size: ", function.data.size(), " bytes\n");

    out.printf(
        "bytecode: %lu instructions (%lu 16-bit instructions, %lu 32-bit instructions); %lu bytes; %d parameter(s); %d local(s); %d callee register(s)\n",
        static_cast<unsigned long>(instructionCount),
        static_cast<unsigned long>(wide16InstructionCount),
        static_cast<unsigned long>(wide32InstructionCount),
        static_cast<unsigned long>(block->instructions().sizeInBytes()),
        block->numArguments(),
        block->numVars(),
        block->numCalleeLocals());

    BytecodeDumper dumper(block, out);
    for (auto it = block->instructions().begin(); it != block->instructions().end(); it += it->size()) {
        dumpWasm(&dumper, it.offset(), it.ptr());
        out.print("\n");
    }

    dumper.dumpConstants();
    dumper.dumpExceptionHandlers();

    out.printf("\n");
}

void BytecodeDumper::dumpConstants()
{
    FunctionCodeBlockGenerator* block = this->block();
    if (!block->constants().isEmpty()) {
        this->m_out.printf("\nConstants:\n");
        unsigned i = 0;
        for (const auto& constant : block->constants()) {
            Type type = block->constantTypes()[i];
            this->m_out.print("   const", i, " : ", type.kind, " = ", formatConstant(type, constant), "\n");
            ++i;
        }
    }
}

void BytecodeDumper::dumpExceptionHandlers()
{
    if (unsigned count = this->block()->numberOfExceptionHandlers()) {
        this->m_out.printf("\nException Handlers:\n");
        unsigned i = 0;
        do {
            const auto& handler = this->block()->exceptionHandler(i);
            this->m_out.printf("\t %d: { start: [%4d] end: [%4d] target: [%4d] tryDepth: [%4d] exceptionIndexOrDelegateTarget: [%4d] } %s\n", i + 1, handler.m_start, handler.m_end, handler.m_target, handler.m_tryDepth, handler.m_exceptionIndexOrDelegateTarget, handler.typeName().characters());
            ++i;
        } while (i < count);
    }
}

CString BytecodeDumper::constantName(VirtualRegister index) const
{
    FunctionCodeBlockGenerator* block = this->block();
    auto value = formatConstant(block->getConstantType(index), block->getConstant(index));
    return toCString(value, "(", VirtualRegister(index), ")");
}

CString BytecodeDumper::formatConstant(Type type, uint64_t constant) const
{
    switch (type.kind) {
    case TypeKind::I32:
        return toCString(static_cast<int32_t>(constant));
    case TypeKind::I64:
        return toCString(constant);
    case TypeKind::F32:
        return toCString(std::bit_cast<float>(static_cast<int32_t>(constant)));
        break;
    case TypeKind::F64:
        return toCString(std::bit_cast<double>(constant));
        break;
    case TypeKind::V128:
        return toCString(constant);
        break;
    default: {
        // This is necessary to handle all cases, since when typed function
        // references are enabled, if type.isFuncref() is true, then
        // isRefType(type) is false (likewise for externref)
        if (isRefType(type) || type.isFuncref() || type.isExternref()) {
            if (JSValue::decode(constant) == jsNull())
                return "null";
            return toCString(RawHex(constant));
        }

        RELEASE_ASSERT_NOT_REACHED();
        return "";
    }
    }
}

CString BytecodeDumper::registerName(VirtualRegister r) const
{
    if (r.isConstant())
        return constantName(r);

    return toCString(r);
}

int BytecodeDumper::outOfLineJumpOffset(WasmInstructionStream::Offset offset) const
{
    return m_block->outOfLineJumpOffset(offset);
}

} // namespace Wasm

#endif // ENABLE(WEBASSEMBLY)
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
