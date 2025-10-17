/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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

#include "AirSpecial.h"
#include "RegisterSet.h"
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 { namespace Air {

// Use this special for constructing a C call. Arg 0 is of course a Special arg that refers to the
// CCallSpecial object. Arg 1 is the callee, and it can be an ImmPtr, a register, or an address. The
// next three args - arg 2, arg 3, and arg 4 - hold the return value GPRs and FPR. The remaining args
// are just the set of argument registers used by this call. For arguments that go to the stack, you
// have to do the grunt work of doing those stack stores. In fact, the only reason why we specify the
// argument registers as arguments to a call is so that the liveness analysis can see that they get
// used here. It would be wrong to automagically report all argument registers as being used because
// if we had a call that didn't pass them, then they'd appear to be live until some clobber point or
// the prologue, whichever happened sooner.

class CCallSpecial final : public Special {
    WTF_MAKE_TZONE_ALLOCATED(CCallSpecial);
public:
    CCallSpecial(bool isSIMDContext);
    ~CCallSpecial() final;

    // You cannot use this register to pass arguments. It just so happens that this register is not
    // used for arguments in the C calling convention. By the way, this is the only thing that causes
    // this special to be specific to C calls.
    static constexpr GPRReg scratchRegister = GPRInfo::nonPreservedNonArgumentGPR0;

private:
    void forEachArg(Inst&, const ScopedLambda<Inst::EachArgCallback>&) final;
    bool isValid(Inst&) final;
    bool admitsStack(Inst&, unsigned argIndex) final;
    bool admitsExtendedOffsetAddr(Inst&, unsigned) final;
    void reportUsedRegisters(Inst&, const RegisterSetBuilder&) final;
    MacroAssembler::Jump generate(Inst&, CCallHelpers&, GenerationContext&) final;
    RegisterSetBuilder extraEarlyClobberedRegs(Inst&) final;
    RegisterSetBuilder extraClobberedRegs(Inst&) final;

    void dumpImpl(PrintStream&) const final;
    void deepDumpImpl(PrintStream&) const final;

    static constexpr unsigned specialArgOffset = 0;
    static constexpr unsigned numSpecialArgs = 1;
    static constexpr unsigned calleeArgOffset = numSpecialArgs;
    static constexpr unsigned numCalleeArgs = 1;
    static constexpr unsigned returnGPArgOffset = numSpecialArgs + numCalleeArgs;
    static constexpr unsigned numReturnGPArgs = 2;
    static constexpr unsigned returnFPArgOffset = numSpecialArgs + numCalleeArgs + numReturnGPArgs;
    static constexpr unsigned numReturnFPArgs = 1;
    static constexpr unsigned argArgOffset =
        numSpecialArgs + numCalleeArgs + numReturnGPArgs + numReturnFPArgs;
    
    RegisterSetBuilder m_clobberedRegs;
    bool m_isSIMDContext { false };
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
