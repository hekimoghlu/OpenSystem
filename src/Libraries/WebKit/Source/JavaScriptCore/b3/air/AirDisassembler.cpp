/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#include "AirDisassembler.h"

#if ENABLE(B3_JIT)

#include "AirBasicBlock.h"
#include "AirCode.h"
#include "AirInst.h"
#include "CCallHelpers.h"
#include "Disassembler.h"
#include "LinkBuffer.h"
#include <wtf/TZoneMallocInlines.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 { namespace Air {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Disassembler);

void Disassembler::startEntrypoint(CCallHelpers& jit)
{
    m_entrypointStart = jit.labelIgnoringWatchpoints();
}

void Disassembler::endEntrypoint(CCallHelpers& jit)
{
    m_entrypointEnd = jit.labelIgnoringWatchpoints();
}

void Disassembler::startLatePath(CCallHelpers& jit)
{
    m_latePathStart = jit.labelIgnoringWatchpoints();
}

void Disassembler::endLatePath(CCallHelpers& jit)
{
    m_latePathEnd = jit.labelIgnoringWatchpoints();
}

void Disassembler::startBlock(BasicBlock* block, CCallHelpers& jit)
{
    UNUSED_PARAM(jit);
    m_blocks.append(block);
}

void Disassembler::addInst(Inst* inst, MacroAssembler::Label start, MacroAssembler::Label end)
{
    auto addResult = m_instToRange.add(inst, std::make_pair(start, end));
    RELEASE_ASSERT(addResult.isNewEntry);
}

void Disassembler::dump(Code& code, PrintStream& out, LinkBuffer& linkBuffer, const char* airPrefix, const char* asmPrefix, const ScopedLambda<void(Inst&)>& doToEachInst)
{
    void* codeStart = linkBuffer.entrypoint<DisassemblyPtrTag>().untaggedPtr();
    void* codeEnd = std::bit_cast<uint8_t*>(codeStart) +  linkBuffer.size();

    auto dumpAsmRange = [&] (CCallHelpers::Label startLabel, CCallHelpers::Label endLabel) {
        RELEASE_ASSERT(startLabel.isSet());
        RELEASE_ASSERT(endLabel.isSet());
        CodeLocationLabel<DisassemblyPtrTag> start = linkBuffer.locationOf<DisassemblyPtrTag>(startLabel);
        CodeLocationLabel<DisassemblyPtrTag> end = linkBuffer.locationOf<DisassemblyPtrTag>(endLabel);
        RELEASE_ASSERT(end.dataLocation<uintptr_t>() >= start.dataLocation<uintptr_t>());
        disassemble(start, end.dataLocation<uintptr_t>() - start.dataLocation<uintptr_t>(), codeStart, codeEnd, asmPrefix, out);
    };

    for (BasicBlock* block : m_blocks) {
        block->dumpHeader(out);
        if (code.isEntrypoint(block))
            dumpAsmRange(m_entrypointStart, m_entrypointEnd);

        for (Inst& inst : *block) {
            doToEachInst(inst);

            out.print(airPrefix);
            inst.dump(out);
            out.print("\n");

            auto iter = m_instToRange.find(&inst);
            if (iter == m_instToRange.end()) {
                RELEASE_ASSERT(&inst == &block->last());
                continue;
            }
            auto pair = iter->value;
            dumpAsmRange(pair.first, pair.second);
        }
        block->dumpFooter(out);
    }

    // FIXME: We could be better about various late paths. We can implement
    // this later if we find a strong use for it.
    out.print(tierName, "# Late paths\n");
    dumpAsmRange(m_latePathStart, m_latePathEnd);

    {
        CodeLocationLabel<DisassemblyPtrTag> start = linkBuffer.locationOf<DisassemblyPtrTag>(m_latePathEnd);
        size_t dumpedSize = start.dataLocation<uintptr_t>() - linkBuffer.entrypoint<DisassemblyPtrTag>().dataLocation<uintptr_t>();
        if (dumpedSize < linkBuffer.size()) {
            out.print(tierName, "# Remaining\n");
            disassemble(start, linkBuffer.size() - dumpedSize, codeStart, codeEnd, asmPrefix, out);
        }
    }
}

} } } // namespace JSC::B3::Air

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
