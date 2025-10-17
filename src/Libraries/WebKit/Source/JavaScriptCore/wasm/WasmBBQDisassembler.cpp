/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#include "WasmBBQDisassembler.h"

#if ENABLE(WEBASSEMBLY_BBQJIT)

#include "Disassembler.h"
#include "LinkBuffer.h"
#include <wtf/HexNumber.h>
#include <wtf/StringPrintStream.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {
namespace Wasm {

ASCIILiteral makeString(PrefixedOpcode prefixedOpcode)
{
    switch (prefixedOpcode.prefixOrOpcode) {
    case OpType::Ext1:
        return makeString(prefixedOpcode.prefixed.ext1Opcode);
    case OpType::ExtSIMD:
        return makeString(prefixedOpcode.prefixed.simdOpcode);
    case OpType::ExtGC:
        return makeString(prefixedOpcode.prefixed.gcOpcode);
    case OpType::ExtAtomic:
        return makeString(prefixedOpcode.prefixed.atomicOpcode);
    default:
        return makeString(prefixedOpcode.prefixOrOpcode);
    }
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(BBQDisassembler);

BBQDisassembler::BBQDisassembler() = default;

BBQDisassembler::~BBQDisassembler() = default;

void BBQDisassembler::dump(PrintStream& out, LinkBuffer& linkBuffer)
{
    m_codeStart = linkBuffer.entrypoint<DisassemblyPtrTag>().untaggedPtr();
    m_codeEnd = std::bit_cast<uint8_t*>(m_codeStart) + linkBuffer.size();

    dumpHeader(out, linkBuffer);
    if (m_labels.isEmpty())
        dumpDisassembly(out, linkBuffer, m_startOfCode, m_endOfCode);
    else {
        dumpDisassembly(out, linkBuffer, m_startOfCode, std::get<0>(m_labels[0]));
        dumpForInstructions(out, linkBuffer, "    ", m_labels, m_endOfOpcode);
        out.print("    (End Of Main Code)\n");
        dumpDisassembly(out, linkBuffer, m_endOfOpcode, m_endOfCode);
    }
}

void BBQDisassembler::dump(LinkBuffer& linkBuffer)
{
    dump(WTF::dataFile(), linkBuffer);
}

void BBQDisassembler::dumpHeader(PrintStream& out, LinkBuffer& linkBuffer)
{
    out.print("   Code at [", RawPointer(linkBuffer.debugAddress()), ", ", RawPointer(static_cast<char*>(linkBuffer.debugAddress()) + linkBuffer.size()), "):\n");
}

Vector<BBQDisassembler::DumpedOp> BBQDisassembler::dumpVectorForInstructions(LinkBuffer& linkBuffer, const char* prefix, Vector<std::tuple<MacroAssembler::Label, PrefixedOpcode, size_t>>& labels, MacroAssembler::Label endLabel)
{
    StringPrintStream out;
    Vector<DumpedOp> result;

    for (unsigned i = 0; i < labels.size();) {
        out.reset();
        auto opcode = std::get<1>(labels[i]);
        auto offset = std::get<2>(labels[i]);
        result.append(DumpedOp { { } });
        out.print(prefix);
        out.println("[", makeString(pad(' ', 8, makeString("0x"_s, hex(offset, 0, Lowercase)))), "] "_s, makeString(opcode));
        unsigned nextIndex = i + 1;
        if (nextIndex >= labels.size()) {
            dumpDisassembly(out, linkBuffer, std::get<0>(labels[i]), endLabel);
            result.last().disassembly = out.toCString();
            return result;
        }
        dumpDisassembly(out, linkBuffer, std::get<0>(labels[i]), std::get<0>(labels[nextIndex]));
        result.last().disassembly = out.toCString();
        i = nextIndex;
    }

    return result;
}

void BBQDisassembler::dumpForInstructions(PrintStream& out, LinkBuffer& linkBuffer, const char* prefix, Vector<std::tuple<MacroAssembler::Label, PrefixedOpcode, size_t>>& labels, MacroAssembler::Label endLabel)
{
    Vector<DumpedOp> dumpedOps = dumpVectorForInstructions(linkBuffer, prefix, labels, endLabel);

    for (unsigned i = 0; i < dumpedOps.size(); ++i)
        out.print(dumpedOps[i].disassembly);
}

void BBQDisassembler::dumpDisassembly(PrintStream& out, LinkBuffer& linkBuffer, MacroAssembler::Label from, MacroAssembler::Label to)
{
    CodeLocationLabel<DisassemblyPtrTag> fromLocation = linkBuffer.locationOf<DisassemblyPtrTag>(from);
    CodeLocationLabel<DisassemblyPtrTag> toLocation = linkBuffer.locationOf<DisassemblyPtrTag>(to);
    disassemble(fromLocation, toLocation.dataLocation<uintptr_t>() - fromLocation.dataLocation<uintptr_t>(), m_codeStart, m_codeEnd, "        ", out);
}

} // namespace Wasm
} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(WEBASSEMBLY_BBQJIT)
