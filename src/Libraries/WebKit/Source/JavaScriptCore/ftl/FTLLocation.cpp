/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
#include "FTLLocation.h"

#if ENABLE(FTL_JIT)

#include "B3ValueRep.h"
#include "FTLSaveRestore.h"
#include "RegisterSet.h"
#include <wtf/DataLog.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace FTL {

using namespace B3;

Location Location::forValueRep(const ValueRep& rep)
{
    switch (rep.kind()) {
    case ValueRep::Register:
        return forRegister(rep.reg(), 0);
    case ValueRep::Stack:
        return forIndirect(GPRInfo::callFrameRegister, rep.offsetFromFP());
    case ValueRep::Constant:
        return forConstant(rep.value());
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return Location();
    }
}

void Location::dump(PrintStream& out) const
{
    out.print("(", kind());
    if (hasReg())
        out.print(", ", reg());
    if (hasOffset())
        out.print(", ", offset());
    if (hasAddend())
        out.print(", ", addend());
    if (hasConstant())
        out.print(", ", constant());
    out.print(")");
}

bool Location::involvesGPR() const
{
    return isGPR() || kind() == Indirect;
}

bool Location::isGPR() const
{
    return kind() == Register && reg().isGPR();
}

GPRReg Location::gpr() const
{
    return reg().gpr();
}

bool Location::isFPR() const
{
    return kind() == Register && reg().isFPR();
}

FPRReg Location::fpr() const
{
    return reg().fpr();
}

void Location::restoreInto(MacroAssembler& jit, char* savedRegisters, GPRReg result, unsigned numFramesToPop) const
{
    if (involvesGPR() && RegisterSetBuilder::stackRegisters().contains(gpr(), IgnoreVectors)) {
        // Make the result GPR contain the appropriate stack register.
        if (numFramesToPop) {
            jit.move(MacroAssembler::framePointerRegister, result);
            
            for (unsigned i = numFramesToPop - 1; i--;)
                jit.loadPtr(MacroAssembler::Address(result), result);
            
            if (gpr() == MacroAssembler::framePointerRegister)
                jit.loadPtr(MacroAssembler::Address(result), result);
            else
                jit.addPtr(MacroAssembler::TrustedImmPtr(sizeof(void*) * 2), result);
        } else
            jit.move(gpr(), result);
    }
    
    if (isGPR()) {
        if (RegisterSetBuilder::stackRegisters().contains(gpr(), IgnoreVectors)) {
            // Already restored into result.
        } else
            jit.load64(savedRegisters + offsetOfGPR(gpr()), result);
        
        if (addend())
            jit.add64(MacroAssembler::TrustedImm32(addend()), result);
        return;
    }
    
    if (isFPR()) {
        jit.load64(savedRegisters + offsetOfFPR(fpr()), result);
        ASSERT(!addend());
        return;
    }
    
    switch (kind()) {
    case Register:
        // B3 used some register that we don't know about!
        dataLog("Unrecognized location: ", *this, "\n");
        RELEASE_ASSERT_NOT_REACHED();
        return;
        
    case Indirect:
        if (RegisterSetBuilder::stackRegisters().contains(gpr(), IgnoreVectors)) {
            // The stack register is already recovered into result.
            jit.load64(MacroAssembler::Address(result, offset()), result);
            return;
        }
        
        jit.load64(savedRegisters + offsetOfGPR(gpr()), result);
        jit.load64(MacroAssembler::Address(result, offset()), result);
        return;
        
    case Constant:
        jit.move(MacroAssembler::TrustedImm64(constant()), result);
        return;
        
    case Unprocessed:
        RELEASE_ASSERT_NOT_REACHED();
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

GPRReg Location::directGPR() const
{
    RELEASE_ASSERT(!addend());
    return gpr();
}

} } // namespace JSC::FTL

namespace WTF {

using namespace JSC::FTL;

void printInternal(PrintStream& out, JSC::FTL::Location::Kind kind)
{
    switch (kind) {
    case Location::Unprocessed:
        out.print("Unprocessed");
        return;
    case Location::Register:
        out.print("Register");
        return;
    case Location::Indirect:
        out.print("Indirect");
        return;
    case Location::Constant:
        out.print("Constant");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(FTL_JIT)
