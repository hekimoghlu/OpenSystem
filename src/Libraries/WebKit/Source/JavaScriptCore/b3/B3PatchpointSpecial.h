/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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

#include "B3StackmapSpecial.h"

namespace JSC { namespace B3 {

// This is a special that recognizes that there are two uses of Patchpoint: Void and and non-Void.
// In the Void case, the syntax of the Air Patch instruction is:
//
//     Patch &patchpoint, args...
//
// Where "args..." are the lowered arguments to the Patchpoint instruction. In the non-Void case
// we will have:
//
//     Patch &patchpoint, result, args...

class PatchpointSpecial final : public StackmapSpecial {
public:
    JS_EXPORT_PRIVATE PatchpointSpecial();
    JS_EXPORT_PRIVATE ~PatchpointSpecial() final;

private:
    void forEachArg(Air::Inst&, const ScopedLambda<Air::Inst::EachArgCallback>&) final;
    bool isValid(Air::Inst&) final;
    bool admitsStack(Air::Inst&, unsigned argIndex) final;
    bool admitsExtendedOffsetAddr(Air::Inst&, unsigned) final;

    // NOTE: the generate method will generate the hidden branch and then register a LatePath that
    // generates the stackmap. Super crazy dude!

    MacroAssembler::Jump generate(Air::Inst&, CCallHelpers&, Air::GenerationContext&) final;
    
    bool isTerminal(Air::Inst&) final;

    void dumpImpl(PrintStream&) const final;
    void deepDumpImpl(PrintStream&) const final;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
