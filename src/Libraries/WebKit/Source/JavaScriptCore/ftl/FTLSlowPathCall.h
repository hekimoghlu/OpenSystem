/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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

#if ENABLE(FTL_JIT)

#include "CCallHelpers.h"
#include "FTLSlowPathCallKey.h"
#include "FTLState.h"

namespace JSC { namespace FTL {

class SlowPathCall {
public:
    SlowPathCall() { }
    
    SlowPathCall(MacroAssembler::Call call, const SlowPathCallKey& key)
        : m_call(call)
        , m_key(key)
    {
    }
    
    MacroAssembler::Call call() const { return m_call; }
    SlowPathCallKey key() const { return m_key; }
    
private:
    MacroAssembler::Call m_call;
    SlowPathCallKey m_key;
};

// This will be an RAII thingy that will set up the necessary stack sizes and offsets and such.
class SlowPathCallContext {
public:
    SlowPathCallContext(ScalarRegisterSet usedRegisters, CCallHelpers&, unsigned numArgs, GPRReg returnRegister, GPRReg indirectCallTargetRegister);
    ~SlowPathCallContext();

    // NOTE: The call that this returns is already going to be linked by the JIT using addLinkTask(),
    // so there is no need for you to link it yourself.
    SlowPathCall makeCall(VM&, CodePtr<CFunctionPtrTag> callTarget);
    SlowPathCall makeCall(VM&, CCallHelpers::Address);

private:
    SlowPathCallKey keyWithTarget(CodePtr<CFunctionPtrTag> callTarget) const;
    SlowPathCallKey keyWithTarget(CCallHelpers::Address) const;
    
    ScalarRegisterSet m_argumentRegisters;
    ScalarRegisterSet m_callingConventionRegisters;
    CCallHelpers& m_jit;
    unsigned m_numArgs;
    GPRReg m_returnRegister;
    size_t m_offsetToSavingArea;
    size_t m_stackBytesNeeded;
    ScalarRegisterSet m_thunkSaveSet;
    size_t m_offset;
};

template<typename... ArgumentTypes>
SlowPathCall callOperation(
    VM& vm, const ScalarRegisterSet& usedRegisters, CCallHelpers& jit, CCallHelpers::JumpList* exceptionTarget,
    CodePtr<CFunctionPtrTag> function, GPRReg resultGPR, ArgumentTypes... arguments)
{
    SlowPathCall call;
    {
        SlowPathCallContext context(usedRegisters, jit, sizeof...(ArgumentTypes) + 1, resultGPR, InvalidGPRReg);
        jit.setupArguments<void(ArgumentTypes...)>(arguments...);
        call = context.makeCall(vm, function);
    }
    if (exceptionTarget)
        exceptionTarget->append(jit.emitExceptionCheck(vm));
    return call;
}

template<typename... ArgumentTypes>
SlowPathCall callOperation(
    VM& vm, const RegisterSetBuilder& usedRegisters, CCallHelpers& jit, CCallHelpers::JumpList* exceptionTarget,
    CodePtr<CFunctionPtrTag> function, GPRReg resultGPR, ArgumentTypes... arguments)
{
    auto regs = usedRegisters.buildScalarRegisterSet();
    return callOperation(vm, regs, jit, exceptionTarget, function, resultGPR, arguments...);
}

template<typename RS, typename... ArgumentTypes>
SlowPathCall callOperation(
    VM& vm, const RS& usedRegisters, CCallHelpers& jit, CallSiteIndex callSiteIndex,
    CCallHelpers::JumpList* exceptionTarget, CodePtr<CFunctionPtrTag> function, GPRReg resultGPR,
    ArgumentTypes... arguments)
{
    if (callSiteIndex) {
        jit.store32(
            CCallHelpers::TrustedImm32(callSiteIndex.bits()),
            CCallHelpers::tagFor(CallFrameSlot::argumentCountIncludingThis));
    }
    return callOperation(vm, usedRegisters, jit, exceptionTarget, function, resultGPR, arguments...);
}

CallSiteIndex callSiteIndexForCodeOrigin(State&, CodeOrigin);

template<typename RS, typename... ArgumentTypes>
SlowPathCall callOperation(
    State& state, const RS& usedRegisters, CCallHelpers& jit, CodeOrigin codeOrigin,
    CCallHelpers::JumpList* exceptionTarget, CodePtr<CFunctionPtrTag> function, GPRReg result, ArgumentTypes... arguments)
{
    return callOperation(
        state.vm(), usedRegisters, jit, callSiteIndexForCodeOrigin(state, codeOrigin), exceptionTarget, function,
        result, arguments...);
}

template<typename... ArgumentTypes>
SlowPathCall callOperation(
    VM& vm, const ScalarRegisterSet& usedRegisters, CCallHelpers& jit, CCallHelpers::JumpList* exceptionTarget,
    CCallHelpers::Address function, GPRReg resultGPR, ArgumentTypes... arguments)
{
    SlowPathCall call;
    {
        SlowPathCallContext context(usedRegisters, jit, sizeof...(ArgumentTypes) + 1, resultGPR, GPRInfo::nonArgGPR0);
        jit.setupArgumentsForIndirectCall<void(ArgumentTypes...)>(function, arguments...);
        call = context.makeCall(vm, CCallHelpers::Address(GPRInfo::nonArgGPR0, function.offset));
    }
    if (exceptionTarget)
        exceptionTarget->append(jit.emitExceptionCheck(vm));
    return call;
}

template<typename... ArgumentTypes>
SlowPathCall callOperation(
    VM& vm, const RegisterSetBuilder& usedRegisters, CCallHelpers& jit, CCallHelpers::JumpList* exceptionTarget,
    CCallHelpers::Address function, GPRReg resultGPR, ArgumentTypes... arguments)
{
    auto regs = usedRegisters.buildScalarRegisterSet();
    return callOperation(vm, regs, jit, exceptionTarget, function, resultGPR, arguments...);
}

template<typename RS, typename... ArgumentTypes>
SlowPathCall callOperation(
    VM& vm, const RS& usedRegisters, CCallHelpers& jit, CallSiteIndex callSiteIndex,
    CCallHelpers::JumpList* exceptionTarget, CCallHelpers::Address function, GPRReg resultGPR,
    ArgumentTypes... arguments)
{
    if (callSiteIndex) {
        jit.store32(
            CCallHelpers::TrustedImm32(callSiteIndex.bits()),
            CCallHelpers::tagFor(CallFrameSlot::argumentCountIncludingThis));
    }
    return callOperation(vm, usedRegisters, jit, exceptionTarget, function, resultGPR, arguments...);
}

CallSiteIndex callSiteIndexForCodeOrigin(State&, CodeOrigin);

template<typename RS, typename... ArgumentTypes>
SlowPathCall callOperation(
    State& state, const RS& usedRegisters, CCallHelpers& jit, CodeOrigin codeOrigin,
    CCallHelpers::JumpList* exceptionTarget, CCallHelpers::Address function, GPRReg result, ArgumentTypes... arguments)
{
    return callOperation(
        state.vm(), usedRegisters, jit, callSiteIndexForCodeOrigin(state, codeOrigin), exceptionTarget, function,
        result, arguments...);
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
