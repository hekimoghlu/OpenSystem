/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#include "JITDisassembler.h"

#if ENABLE(JIT)

#include "CodeBlock.h"
#include "CodeBlockWithJITType.h"
#include "Disassembler.h"
#include "JSCellInlines.h"
#include "LinkBuffer.h"
#include "ProfilerCompilation.h"
#include <wtf/StringPrintStream.h>
#include <wtf/TZoneMallocInlines.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(JITDisassembler);

JITDisassembler::JITDisassembler(CodeBlock *codeBlock)
    : m_codeBlock(codeBlock)
    , m_labelForBytecodeIndexInMainPath(codeBlock->instructions().size())
    , m_labelForBytecodeIndexInSlowPath(codeBlock->instructions().size())
{
}

JITDisassembler::~JITDisassembler() = default;

void JITDisassembler::dump(PrintStream& out, LinkBuffer& linkBuffer)
{
    m_codeStart = linkBuffer.entrypoint<DisassemblyPtrTag>().untaggedPtr();
    m_codeEnd = std::bit_cast<uint8_t*>(m_codeStart) + linkBuffer.size();

    dumpHeader(out, linkBuffer);
    dumpDisassembly(out, linkBuffer, m_startOfCode, m_labelForBytecodeIndexInMainPath[0]);
    
    dumpForInstructions(out, linkBuffer, "    ", m_labelForBytecodeIndexInMainPath, firstSlowLabel());
    out.print("    (End Of Main Path)\n");
    dumpForInstructions(out, linkBuffer, "    (S) ", m_labelForBytecodeIndexInSlowPath, m_endOfSlowPath);
    out.print("    (End Of Slow Path)\n");

    dumpDisassembly(out, linkBuffer, m_endOfSlowPath, m_endOfCode);
}

void JITDisassembler::dump(LinkBuffer& linkBuffer)
{
    dump(WTF::dataFile(), linkBuffer);
}

void JITDisassembler::reportToProfiler(Profiler::Compilation* compilation, LinkBuffer& linkBuffer)
{
    StringPrintStream out;
    
    dumpHeader(out, linkBuffer);
    compilation->addDescription(Profiler::CompiledBytecode(Profiler::OriginStack(), out.toCString()));
    out.reset();
    dumpDisassembly(out, linkBuffer, m_startOfCode, m_labelForBytecodeIndexInMainPath[0]);
    compilation->addDescription(Profiler::CompiledBytecode(Profiler::OriginStack(), out.toCString()));
    
    reportInstructions(compilation, linkBuffer, "    ", m_labelForBytecodeIndexInMainPath, firstSlowLabel());
    compilation->addDescription(Profiler::CompiledBytecode(Profiler::OriginStack(), "    (End Of Main Path)\n"));
    reportInstructions(compilation, linkBuffer, "    (S) ", m_labelForBytecodeIndexInSlowPath, m_endOfSlowPath);
    compilation->addDescription(Profiler::CompiledBytecode(Profiler::OriginStack(), "    (End Of Slow Path)\n"));
    out.reset();
    dumpDisassembly(out, linkBuffer, m_endOfSlowPath, m_endOfCode);
    compilation->addDescription(Profiler::CompiledBytecode(Profiler::OriginStack(), out.toCString()));
}

void JITDisassembler::dumpHeader(PrintStream& out, LinkBuffer& linkBuffer)
{
    out.print("Generated Baseline JIT code for ", CodeBlockWithJITType(m_codeBlock, JITType::BaselineJIT), ", instructions size = ", m_codeBlock->instructionsSize(), "\n");
    out.print("   Source: ", m_codeBlock->sourceCodeOnOneLine(), "\n");
    out.print("   Code at [", RawPointer(linkBuffer.debugAddress()), ", ", RawPointer(static_cast<char*>(linkBuffer.debugAddress()) + linkBuffer.size()), "):\n");
}

MacroAssembler::Label JITDisassembler::firstSlowLabel()
{
    MacroAssembler::Label firstSlowLabel;
    for (unsigned i = 0; i < m_labelForBytecodeIndexInSlowPath.size(); ++i) {
        if (m_labelForBytecodeIndexInSlowPath[i].isSet()) {
            firstSlowLabel = m_labelForBytecodeIndexInSlowPath[i];
            break;
        }
    }
    return firstSlowLabel.isSet() ? firstSlowLabel : m_endOfSlowPath;
}

Vector<JITDisassembler::DumpedOp> JITDisassembler::dumpVectorForInstructions(LinkBuffer& linkBuffer, const char* prefix, Vector<MacroAssembler::Label>& labels, MacroAssembler::Label endLabel)
{
    StringPrintStream out;
    Vector<DumpedOp> result;
    
    for (unsigned i = 0; i < labels.size();) {
        if (!labels[i].isSet()) {
            i++;
            continue;
        }
        out.reset();
        result.append(DumpedOp());
        result.last().bytecodeIndex = BytecodeIndex(i);
        out.print(prefix);
        m_codeBlock->dumpBytecode(out, i);
        for (unsigned nextIndex = i + 1; ; nextIndex++) {
            if (nextIndex >= labels.size()) {
                dumpDisassembly(out, linkBuffer, labels[i], endLabel);
                result.last().disassembly = out.toCString();
                return result;
            }
            if (labels[nextIndex].isSet()) {
                dumpDisassembly(out, linkBuffer, labels[i], labels[nextIndex]);
                result.last().disassembly = out.toCString();
                i = nextIndex;
                break;
            }
        }
    }
    
    return result;
}

void JITDisassembler::dumpForInstructions(PrintStream& out, LinkBuffer& linkBuffer, const char* prefix, Vector<MacroAssembler::Label>& labels, MacroAssembler::Label endLabel)
{
    Vector<DumpedOp> dumpedOps = dumpVectorForInstructions(linkBuffer, prefix, labels, endLabel);
    
    for (unsigned i = 0; i < dumpedOps.size(); ++i)
        out.print(dumpedOps[i].disassembly);
}

void JITDisassembler::reportInstructions(Profiler::Compilation* compilation, LinkBuffer& linkBuffer, const char* prefix, Vector<MacroAssembler::Label>& labels, MacroAssembler::Label endLabel)
{
    Vector<DumpedOp> dumpedOps = dumpVectorForInstructions(linkBuffer, prefix, labels, endLabel);
    
    for (unsigned i = 0; i < dumpedOps.size(); ++i) {
        compilation->addDescription(
            Profiler::CompiledBytecode(
                Profiler::OriginStack(Profiler::Origin(compilation->bytecodes(), dumpedOps[i].bytecodeIndex)),
                dumpedOps[i].disassembly));
    }
}

void JITDisassembler::dumpDisassembly(PrintStream& out, LinkBuffer& linkBuffer, MacroAssembler::Label from, MacroAssembler::Label to)
{
    CodeLocationLabel<DisassemblyPtrTag> fromLocation = linkBuffer.locationOf<DisassemblyPtrTag>(from);
    CodeLocationLabel<DisassemblyPtrTag> toLocation = linkBuffer.locationOf<DisassemblyPtrTag>(to);
    disassemble(fromLocation, toLocation.dataLocation<uintptr_t>() - fromLocation.dataLocation<uintptr_t>(), m_codeStart, m_codeEnd, "        ", out);
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(JIT)
