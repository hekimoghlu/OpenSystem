/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

#if ENABLE(B3_JIT)

#include "MacroAssembler.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class CCallHelpers;
class LinkBuffer;

namespace B3 { namespace Air {

class BasicBlock;
class Code;
struct Inst;

class Disassembler {
    WTF_MAKE_TZONE_ALLOCATED(Disassembler);
public:
    Disassembler() = default;

    void startEntrypoint(CCallHelpers&);
    void endEntrypoint(CCallHelpers&);
    void startLatePath(CCallHelpers&);
    void endLatePath(CCallHelpers&);
    void startBlock(BasicBlock*, CCallHelpers&);
    void addInst(Inst*, MacroAssembler::Label, MacroAssembler::Label);

    void dump(Code&, PrintStream&, LinkBuffer&, const char* airPrefix, const char* asmPrefix, const WTF::ScopedLambda<void(Inst&)>& doToEachInst);

private:
    UncheckedKeyHashMap<Inst*, std::pair<MacroAssembler::Label, MacroAssembler::Label>> m_instToRange;
    Vector<BasicBlock*> m_blocks;
    MacroAssembler::Label m_entrypointStart;
    MacroAssembler::Label m_entrypointEnd;
    MacroAssembler::Label m_latePathStart;
    MacroAssembler::Label m_latePathEnd;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
