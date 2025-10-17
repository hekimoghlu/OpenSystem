/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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

#if ENABLE(JIT)

#include "MacroAssembler.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/CString.h>


namespace JSC {

class LinkBuffer;

namespace Yarr {

class YarrCodeBlock;

class YarrJITInfo {
public:
    virtual ~YarrJITInfo() { };
    virtual const char* variant() = 0;
    virtual unsigned opCount() = 0;
    virtual void dumpPatternString(PrintStream&) = 0;
    virtual int dumpFor(PrintStream&, unsigned) = 0;
};

class YarrDisassembler {
    WTF_MAKE_TZONE_ALLOCATED(YarrDisassembler);
public:
    YarrDisassembler(YarrJITInfo*);
    ~YarrDisassembler();

    void setStartOfCode(MacroAssembler::Label label) { m_startOfCode = label; }
    void setForGenerate(unsigned opIndex, MacroAssembler::Label label)
    {
        m_labelForGenerateYarrOp[opIndex] = label;
    }

    void setForBacktrack(unsigned opIndex, MacroAssembler::Label label)
    {
        m_labelForBacktrackYarrOp[opIndex] = label;
    }

    void setEndOfGenerate(MacroAssembler::Label label) { m_endOfGenerate = label; }
    void setEndOfBacktrack(MacroAssembler::Label label) { m_endOfBacktrack = label; }
    void setEndOfCode(MacroAssembler::Label label) { m_endOfCode = label; }

    void dump(LinkBuffer&);
    void dump(PrintStream&, LinkBuffer&);

private:
    enum class VectorOrder {
        IterateForward,
        IterateReverse
    };

    void dumpHeader(PrintStream&, LinkBuffer&);
    MacroAssembler::Label firstSlowLabel();

    struct DumpedOp {
        unsigned index;
        CString disassembly;
    };

    const char* indentString(unsigned);
    const char* indentString()
    {
        return indentString(m_indentLevel);
    }

    Vector<DumpedOp> dumpVectorForInstructions(LinkBuffer&, Vector<MacroAssembler::Label>& labels, MacroAssembler::Label endLabel, YarrDisassembler::VectorOrder vectorOrder = VectorOrder::IterateForward);

    void dumpForInstructions(PrintStream&, LinkBuffer&, Vector<MacroAssembler::Label>& labels, MacroAssembler::Label endLabel, YarrDisassembler::VectorOrder vectorOrder = VectorOrder::IterateForward);

    void dumpDisassembly(PrintStream&, const char* prefix, LinkBuffer&, MacroAssembler::Label from, MacroAssembler::Label to);

    YarrJITInfo* m_jitInfo;
    MacroAssembler::Label m_startOfCode;
    Vector<MacroAssembler::Label> m_labelForGenerateYarrOp;
    Vector<MacroAssembler::Label> m_labelForBacktrackYarrOp;
    MacroAssembler::Label m_endOfGenerate;
    MacroAssembler::Label m_endOfBacktrack;
    MacroAssembler::Label m_endOfCode;
    void* m_codeStart { nullptr };
    void* m_codeEnd { nullptr };
    unsigned m_indentLevel { 0 };
};

}} // namespace Yarr namespace JSC

#endif // ENABLE(JIT)
