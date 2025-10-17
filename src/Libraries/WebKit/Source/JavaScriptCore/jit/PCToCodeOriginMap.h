/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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

#include "CodeOrigin.h"
#include "MacroAssembler.h"
#include "VM.h"
#include <wtf/Vector.h>

namespace JSC {

#if ENABLE(FTL_JIT) || ENABLE(WEBASSEMBLY_OMGJIT)
namespace B3 {
class PCToOriginMap;
}
#endif

class LinkBuffer;
class PCToCodeOriginMapBuilder;

class PCToCodeOriginMapBuilder {
    WTF_MAKE_TZONE_NON_HEAP_ALLOCATABLE(PCToCodeOriginMapBuilder);
    WTF_MAKE_NONCOPYABLE(PCToCodeOriginMapBuilder);
    friend class PCToCodeOriginMap;

public:
    PCToCodeOriginMapBuilder(VM&);
    PCToCodeOriginMapBuilder(PCToCodeOriginMapBuilder&& other);
    PCToCodeOriginMapBuilder(bool shouldBuildMapping)
        : m_shouldBuildMapping(shouldBuildMapping)
    { }

#if ENABLE(FTL_JIT)
    enum JSTag { JSCodeOriginMap };
    PCToCodeOriginMapBuilder(JSTag, VM&, const B3::PCToOriginMap&);
#endif

#if ENABLE(WEBASSEMBLY_OMGJIT)
    enum WasmTag { WasmCodeOriginMap };
    PCToCodeOriginMapBuilder(WasmTag, const B3::PCToOriginMap&);
#endif

    void appendItem(MacroAssembler::Label label, const CodeOrigin& origin)
    {
        if (!m_shouldBuildMapping)
            return;
        appendItemSlow(label, origin);
    }
    static CodeOrigin defaultCodeOrigin() { return CodeOrigin(BytecodeIndex(0)); }

    bool didBuildMapping() const { return m_shouldBuildMapping; }

private:
    void appendItemSlow(MacroAssembler::Label, const CodeOrigin&);

    struct CodeRange {
        MacroAssembler::Label start;
        MacroAssembler::Label end;
        CodeOrigin codeOrigin;
    };

    Vector<CodeRange> m_codeRanges;
    bool m_shouldBuildMapping;
};

// FIXME: <rdar://problem/39436658>
class PCToCodeOriginMap {
    WTF_MAKE_TZONE_ALLOCATED(PCToCodeOriginMap);
    WTF_MAKE_NONCOPYABLE(PCToCodeOriginMap);
public:
    PCToCodeOriginMap(PCToCodeOriginMapBuilder&&, LinkBuffer&);
    ~PCToCodeOriginMap();

    std::optional<CodeOrigin> findPC(void* pc) const;

    double memorySize();

private:
    size_t m_compressedPCBufferSize;
    size_t m_compressedCodeOriginsSize;
    uint8_t* m_compressedPCs;
    uint8_t* m_compressedCodeOrigins;
    uintptr_t m_pcRangeStart;
    uintptr_t m_pcRangeEnd;
};

} // namespace JSC

#endif // ENABLE(JIT)
