/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#include "YarrDisassembler.h"

#if ENABLE(JIT)

#include "Disassembler.h"
#include "LinkBuffer.h"
#include <wtf/StringPrintStream.h>
#include <wtf/TZoneMallocInlines.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace Yarr {

WTF_MAKE_TZONE_ALLOCATED_IMPL(YarrDisassembler);

static constexpr char s_spaces[] = "                        ";
static constexpr unsigned s_maxIndent = sizeof(s_spaces) - 1;

const char* YarrDisassembler::indentString(unsigned level)
{
    unsigned indent = 6 + level * 2;
    indent = std::min(indent, s_maxIndent);

    return s_spaces + s_maxIndent - indent;
}

YarrDisassembler::YarrDisassembler(YarrJITInfo* yarrJITInfo)
    : m_jitInfo(yarrJITInfo)
    , m_labelForGenerateYarrOp(yarrJITInfo->opCount())
    , m_labelForBacktrackYarrOp(yarrJITInfo->opCount())
{
}

YarrDisassembler::~YarrDisassembler() = default;

void YarrDisassembler::dump(PrintStream& out, LinkBuffer& linkBuffer)
{
    m_codeStart = linkBuffer.entrypoint<DisassemblyPtrTag>().untaggedPtr();
    m_codeEnd = std::bit_cast<uint8_t*>(m_codeStart) + linkBuffer.size();

    dumpHeader(out, linkBuffer);
    dumpDisassembly(out, indentString(), linkBuffer, m_startOfCode, m_labelForGenerateYarrOp[0]);

    out.print("     == Matching ==\n");
    dumpForInstructions(out, linkBuffer, m_labelForGenerateYarrOp, m_endOfGenerate);
    out.print("     == Backtracking ==\n");
    dumpForInstructions(out, linkBuffer, m_labelForBacktrackYarrOp, m_endOfBacktrack, VectorOrder::IterateReverse);

    if (!(m_endOfBacktrack == m_endOfCode)) {
        out.print("     == Helpers ==\n");

        dumpDisassembly(out, indentString(), linkBuffer, m_endOfBacktrack, m_endOfCode);
    }

    linkBuffer.didAlreadyDisassemble();
}

void YarrDisassembler::dump(LinkBuffer& linkBuffer)
{
    dump(WTF::dataFile(), linkBuffer);
}

void YarrDisassembler::dumpHeader(PrintStream& out, LinkBuffer& linkBuffer)
{
    out.print("Generated JIT code for ", m_jitInfo->variant(), " ");
    m_jitInfo->dumpPatternString(out);
    out.print(":\n");
    out.print("    Code at [", RawPointer(linkBuffer.debugAddress()), ", ", RawPointer(static_cast<char*>(linkBuffer.debugAddress()) + linkBuffer.size()), "):\n");
}

Vector<YarrDisassembler::DumpedOp> YarrDisassembler::dumpVectorForInstructions(LinkBuffer& linkBuffer, Vector<MacroAssembler::Label>& labels, MacroAssembler::Label endLabel, YarrDisassembler::VectorOrder vectorOrder)
{
    StringPrintStream out;
    Vector<DumpedOp> result;

    unsigned directionBias = (vectorOrder == VectorOrder::IterateForward) ? 0 : labels.size() - 1;

    auto realIndex = [&](unsigned rawIndex) {
        if (directionBias)
            return directionBias - rawIndex;
        return rawIndex;
    };

    for (unsigned i = 0; i < labels.size();) {
        if (!labels[realIndex(i)].isSet()) {
            i++;
            continue;
        }
        out.reset();
        result.append(DumpedOp());
        result.last().index = realIndex(i);

        int delta = m_jitInfo->dumpFor(out, realIndex(i));
        m_indentLevel += (vectorOrder == VectorOrder::IterateForward) ? delta : -delta;

        for (unsigned nextIndex = i + 1; ; nextIndex++) {
            if (nextIndex >= labels.size()) {
                dumpDisassembly(out, indentString(), linkBuffer, labels[realIndex(i)], endLabel);
                result.last().disassembly = out.toCString();
                return result;
            }
            if (labels[realIndex(nextIndex)].isSet()) {
                dumpDisassembly(out, indentString(), linkBuffer, labels[realIndex(i)], labels[realIndex(nextIndex)]);
                result.last().disassembly = out.toCString();
                i = nextIndex;
                break;
            }
        }
    }

    return result;
}

void YarrDisassembler::dumpForInstructions(PrintStream& out, LinkBuffer& linkBuffer, Vector<MacroAssembler::Label>& labels, MacroAssembler::Label endLabel, YarrDisassembler::VectorOrder vectorOrder)
{
    Vector<DumpedOp> dumpedOps = dumpVectorForInstructions(linkBuffer, labels, endLabel, vectorOrder);

    for (unsigned i = 0; i < dumpedOps.size(); ++i)
        out.print(dumpedOps[i].disassembly);
}

void YarrDisassembler::dumpDisassembly(PrintStream& out, const char* prefix, LinkBuffer& linkBuffer, MacroAssembler::Label from, MacroAssembler::Label to)
{
    CodeLocationLabel<DisassemblyPtrTag> fromLocation = linkBuffer.locationOf<DisassemblyPtrTag>(from);
    CodeLocationLabel<DisassemblyPtrTag> toLocation = linkBuffer.locationOf<DisassemblyPtrTag>(to);
    disassemble(fromLocation, toLocation.dataLocation<uintptr_t>() - fromLocation.dataLocation<uintptr_t>(), m_codeStart, m_codeEnd, prefix, out);
}

}} // namespace Yarr namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(JIT)
