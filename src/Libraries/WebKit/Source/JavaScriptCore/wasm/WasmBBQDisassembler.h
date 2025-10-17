/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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

#if ENABLE(WEBASSEMBLY_BBQJIT)

#include "BytecodeIndex.h"
#include "MacroAssembler.h"
#include "WasmOpcodeOrigin.h"
#include "WasmOps.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/CString.h>

namespace JSC {

class LinkBuffer;

namespace Wasm {

class BBQCallee;

struct PrefixedOpcode {
    OpType prefixOrOpcode;
    union {
        Ext1OpType ext1Opcode;
        ExtAtomicOpType atomicOpcode;
        ExtSIMDOpType simdOpcode;
        ExtGCOpType gcOpcode;
    } prefixed;

    inline explicit PrefixedOpcode(OpType opcode)
    {
        switch (opcode) {
        default:
            prefixOrOpcode = opcode;
            break;
        case OpType::ExtGC:
        case OpType::Ext1:
        case OpType::ExtAtomic:
        case OpType::ExtSIMD:
            RELEASE_ASSERT_NOT_REACHED();
        }
    }

    inline explicit PrefixedOpcode(OpType prefix, uint32_t opcode)
    {
        prefixOrOpcode = prefix;
        switch (prefix) {
        case OpType::Ext1:
            prefixed.ext1Opcode = static_cast<Ext1OpType>(opcode);
            break;
        case OpType::ExtSIMD:
            prefixed.simdOpcode = static_cast<ExtSIMDOpType>(opcode);
            break;
        case OpType::ExtGC:
            prefixed.gcOpcode = static_cast<ExtGCOpType>(opcode);
            break;
        case OpType::ExtAtomic:
            prefixed.atomicOpcode = static_cast<ExtAtomicOpType>(opcode);
            break;
        default:
            RELEASE_ASSERT_NOT_REACHED_WITH_MESSAGE("Expected a valid WASM opcode prefix.");
        }
    }
};

ASCIILiteral makeString(PrefixedOpcode);

class BBQDisassembler {
    WTF_MAKE_TZONE_ALLOCATED(BBQDisassembler);
public:
    BBQDisassembler();
    ~BBQDisassembler();

    void setStartOfCode(MacroAssembler::Label label) { m_startOfCode = label; }
    void setOpcode(MacroAssembler::Label label, PrefixedOpcode opcode, size_t offset)
    {
        m_labels.append(std::tuple { label, opcode, offset });
    }
    void setEndOfOpcode(MacroAssembler::Label label) { m_endOfOpcode = label; }
    void setEndOfCode(MacroAssembler::Label label) { m_endOfCode = label; }

    void dump(LinkBuffer&);
    void dump(PrintStream&, LinkBuffer&);

private:
    void dumpHeader(PrintStream&, LinkBuffer&);

    struct DumpedOp {
        CString disassembly;
    };
    Vector<DumpedOp> dumpVectorForInstructions(LinkBuffer&, const char* prefix, Vector<std::tuple<MacroAssembler::Label, PrefixedOpcode, size_t>>& labels, MacroAssembler::Label endLabel);

    void dumpForInstructions(PrintStream&, LinkBuffer&, const char* prefix, Vector<std::tuple<MacroAssembler::Label, PrefixedOpcode, size_t>>& labels, MacroAssembler::Label endLabel);
    void dumpDisassembly(PrintStream&, LinkBuffer&, MacroAssembler::Label from, MacroAssembler::Label to);

    MacroAssembler::Label m_startOfCode;
    Vector<std::tuple<MacroAssembler::Label, PrefixedOpcode, size_t>> m_labels;
    MacroAssembler::Label m_endOfOpcode;
    MacroAssembler::Label m_endOfCode;
    void* m_codeStart { nullptr };
    void* m_codeEnd { nullptr };
};

} // namespace Wasm
} // namespace JSC

#endif // ENABLE(WEBASSEMBLY_BBQJIT)
