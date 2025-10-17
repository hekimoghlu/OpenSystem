/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include "SlowPathCall.h"

#if ENABLE(JIT)

#include "CCallHelpers.h"
#include "JITInlines.h"
#include "JITThunks.h"
#include "ThunkGenerators.h"
#include "VM.h"

namespace JSC {

namespace {
    constexpr GPRReg bytecodeOffsetGPR = JIT::argumentGPR3;
}

void JITSlowPathCall::call()
{
    VM& vm = m_jit->vm();
    uint32_t bytecodeOffset = m_jit->m_bytecodeIndex.offset();
    ASSERT(BytecodeIndex(bytecodeOffset) == m_jit->m_bytecodeIndex);

    m_jit->move(JIT::TrustedImm32(bytecodeOffset), bytecodeOffsetGPR);
    m_jit->nearCallThunk(CodeLocationLabel { vm.jitStubs->ctiSlowPathFunctionStub(vm, m_slowPathFunction).retaggedCode<NoPtrTag>() });
}

MacroAssemblerCodeRef<JITThunkPtrTag> JITSlowPathCall::generateThunk(VM& vm, SlowPathFunction slowPathFunction)
{
    CCallHelpers jit;

    jit.emitCTIThunkPrologue();

    // Call slow operation
    jit.store32(bytecodeOffsetGPR, CCallHelpers::tagFor(CallFrameSlot::argumentCountIncludingThis));
    jit.prepareCallOperation(vm);

    constexpr GPRReg callFrameArgGPR = GPRInfo::argumentGPR0;
    constexpr GPRReg pcArgGPR = GPRInfo::argumentGPR1;
    static_assert(noOverlap(callFrameArgGPR, pcArgGPR, bytecodeOffsetGPR));

    jit.move(GPRInfo::callFrameRegister, callFrameArgGPR);
    jit.loadPtr(CCallHelpers::addressFor(CallFrameSlot::codeBlock), pcArgGPR);
    jit.loadPtr(CCallHelpers::Address(pcArgGPR, CodeBlock::offsetOfInstructionsRawPointer()), pcArgGPR);
    jit.addPtr(bytecodeOffsetGPR, pcArgGPR);

    jit.callOperation<OperationPtrTag>(slowPathFunction);

    jit.emitCTIThunkEpilogue();

    // Tail call to exception check thunk
    jit.jumpThunk(CodeLocationLabel(vm.getCTIStub(CommonJITThunkID::CheckException).retaggedCode<NoPtrTag>()));

    LinkBuffer patchBuffer(jit, GLOBAL_THUNK_ID, LinkBuffer::Profile::ExtraCTIThunk);
    return FINALIZE_THUNK(patchBuffer, JITThunkPtrTag, "SlowPathCall"_s, "SlowPathCall");
}

} // namespace JSC

#endif // ENABLE(JIT)
