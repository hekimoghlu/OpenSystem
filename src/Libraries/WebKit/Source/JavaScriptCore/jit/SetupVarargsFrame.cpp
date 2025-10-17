/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#include "SetupVarargsFrame.h"

#if ENABLE(JIT)

#include "Interpreter.h"
#include "JSCJSValueInlines.h"
#include "StackAlignment.h"

namespace JSC {

void emitSetVarargsFrame(CCallHelpers& jit, GPRReg lengthGPR, bool lengthIncludesThis, GPRReg numUsedSlotsGPR, GPRReg resultGPR)
{
    jit.move(numUsedSlotsGPR, resultGPR);
    // We really want to make sure the size of the new call frame is a multiple of
    // stackAlignmentRegisters(), however it is easier to accomplish this by
    // rounding numUsedSlotsGPR to the next multiple of stackAlignmentRegisters().
    // Together with the rounding below, we will assure that the new call frame is
    // located on a stackAlignmentRegisters() boundary and a multiple of
    // stackAlignmentRegisters() in size.
    jit.addPtr(CCallHelpers::TrustedImm32(stackAlignmentRegisters() - 1), resultGPR);
    jit.andPtr(CCallHelpers::TrustedImm32(~(stackAlignmentRegisters() - 1)), resultGPR);

    jit.addPtr(lengthGPR, resultGPR);
    jit.addPtr(CCallHelpers::TrustedImm32(CallFrame::headerSizeInRegisters + (lengthIncludesThis ? 0 : 1)), resultGPR);
    
    // resultGPR now has the required frame size in Register units
    // Round resultGPR to next multiple of stackAlignmentRegisters()
    jit.addPtr(CCallHelpers::TrustedImm32(stackAlignmentRegisters() - 1), resultGPR);
    jit.andPtr(CCallHelpers::TrustedImm32(~(stackAlignmentRegisters() - 1)), resultGPR);
    
    // Now resultGPR has the right stack frame offset in Register units.
    jit.negPtr(resultGPR);
    jit.getEffectiveAddress(CCallHelpers::BaseIndex(GPRInfo::callFrameRegister, resultGPR, CCallHelpers::TimesEight), resultGPR);
}

static void emitSetupVarargsFrameFastCase(VM& vm, CCallHelpers& jit, GPRReg numUsedSlotsGPR, GPRReg scratchGPR1, GPRReg scratchGPR2, GPRReg scratchGPR3, ValueRecovery argCountRecovery, VirtualRegister firstArgumentReg, unsigned firstVarArgOffset, CCallHelpers::JumpList& slowCase)
{
    CCallHelpers::JumpList end;
    
    if (argCountRecovery.isConstant()) {
        // FIXME: We could constant-fold a lot of the computation below in this case.
        // https://bugs.webkit.org/show_bug.cgi?id=141486
        jit.move(CCallHelpers::TrustedImm32(argCountRecovery.constant().asInt32()), scratchGPR1);
    } else
        jit.load32(CCallHelpers::payloadFor(argCountRecovery.virtualRegister()), scratchGPR1);
    if (firstVarArgOffset) {
        CCallHelpers::Jump sufficientArguments = jit.branch32(CCallHelpers::GreaterThan, scratchGPR1, CCallHelpers::TrustedImm32(firstVarArgOffset + 1));
        jit.move(CCallHelpers::TrustedImm32(1), scratchGPR1);
        CCallHelpers::Jump endVarArgs = jit.jump();
        sufficientArguments.link(&jit);
        jit.sub32(CCallHelpers::TrustedImm32(firstVarArgOffset), scratchGPR1);
        endVarArgs.link(&jit);
    }
    slowCase.append(jit.branch32(CCallHelpers::Above, scratchGPR1, CCallHelpers::TrustedImm32(JSC::maxArguments + 1)));
    
    emitSetVarargsFrame(jit, scratchGPR1, true, numUsedSlotsGPR, scratchGPR2);

    slowCase.append(jit.branchPtr(CCallHelpers::Above, scratchGPR2, GPRInfo::callFrameRegister));
    slowCase.append(jit.branchPtr(CCallHelpers::Above, CCallHelpers::AbsoluteAddress(vm.addressOfSoftStackLimit()), scratchGPR2));

    // Before touching stack values, we should update the stack pointer to protect them from signal stack.
    jit.addPtr(CCallHelpers::TrustedImm32(sizeof(CallerFrameAndPC)), scratchGPR2, CCallHelpers::stackPointerRegister);

    // Initialize ArgumentCount.
    jit.store32(scratchGPR1, CCallHelpers::Address(scratchGPR2, CallFrameSlot::argumentCountIncludingThis * static_cast<int>(sizeof(Register)) + PayloadOffset));

    // Copy arguments.
    jit.signExtend32ToPtr(scratchGPR1, scratchGPR1);
    CCallHelpers::Jump done = jit.branchSubPtr(CCallHelpers::Zero, CCallHelpers::TrustedImm32(1), scratchGPR1);
    // scratchGPR1: argumentCount

    CCallHelpers::Label copyLoop = jit.label();
    int argOffset = (firstArgumentReg.offset() - 1 + firstVarArgOffset) * static_cast<int>(sizeof(Register));
#if USE(JSVALUE64)
    jit.load64(CCallHelpers::BaseIndex(GPRInfo::callFrameRegister, scratchGPR1, CCallHelpers::TimesEight, argOffset), scratchGPR3);
    jit.store64(scratchGPR3, CCallHelpers::BaseIndex(scratchGPR2, scratchGPR1, CCallHelpers::TimesEight, CallFrame::thisArgumentOffset() * static_cast<int>(sizeof(Register))));
#else // USE(JSVALUE64), so this begins the 32-bit case
    jit.load32(CCallHelpers::BaseIndex(GPRInfo::callFrameRegister, scratchGPR1, CCallHelpers::TimesEight, argOffset + TagOffset), scratchGPR3);
    jit.store32(scratchGPR3, CCallHelpers::BaseIndex(scratchGPR2, scratchGPR1, CCallHelpers::TimesEight, CallFrame::thisArgumentOffset() * static_cast<int>(sizeof(Register)) + TagOffset));
    jit.load32(CCallHelpers::BaseIndex(GPRInfo::callFrameRegister, scratchGPR1, CCallHelpers::TimesEight, argOffset + PayloadOffset), scratchGPR3);
    jit.store32(scratchGPR3, CCallHelpers::BaseIndex(scratchGPR2, scratchGPR1, CCallHelpers::TimesEight, CallFrame::thisArgumentOffset() * static_cast<int>(sizeof(Register)) + PayloadOffset));
#endif // USE(JSVALUE64), end of 32-bit case
    jit.branchSubPtr(CCallHelpers::NonZero, CCallHelpers::TrustedImm32(1), scratchGPR1).linkTo(copyLoop, &jit);
    
    done.link(&jit);
}

void emitSetupVarargsFrameFastCase(VM& vm, CCallHelpers& jit, GPRReg numUsedSlotsGPR, GPRReg scratchGPR1, GPRReg scratchGPR2, GPRReg scratchGPR3, InlineCallFrame* inlineCallFrame, unsigned firstVarArgOffset, CCallHelpers::JumpList& slowCase)
{
    ValueRecovery argumentCountRecovery;
    VirtualRegister firstArgumentReg;
    if (inlineCallFrame) {
        if (inlineCallFrame->isVarargs()) {
            argumentCountRecovery = ValueRecovery::displacedInJSStack(
                inlineCallFrame->argumentCountRegister, DataFormatInt32);
        } else {
            argumentCountRecovery = ValueRecovery::constant(jsNumber(inlineCallFrame->argumentCountIncludingThis));
        }
        if (inlineCallFrame->m_argumentsWithFixup.size() > 1)
            firstArgumentReg = inlineCallFrame->m_argumentsWithFixup[1].virtualRegister();
        else
            firstArgumentReg = VirtualRegister(0);
    } else {
        argumentCountRecovery = ValueRecovery::displacedInJSStack(
            CallFrameSlot::argumentCountIncludingThis, DataFormatInt32);
        firstArgumentReg = VirtualRegister(CallFrame::argumentOffset(0));
    }
    emitSetupVarargsFrameFastCase(vm, jit, numUsedSlotsGPR, scratchGPR1, scratchGPR2, scratchGPR3, argumentCountRecovery, firstArgumentReg, firstVarArgOffset, slowCase);
}

} // namespace JSC

#endif // ENABLE(JIT)

