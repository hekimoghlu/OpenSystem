/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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

#if ENABLE(DFG_JIT)

#include "DFGOperations.h"
#include "DFGSlowPathGenerator.h"
#include "DFGSpeculativeJIT.h"
#include <wtf/Vector.h>

namespace JSC { namespace DFG {

class SaneStringGetByValSlowPathGenerator final : public JumpingSlowPathGenerator<MacroAssembler::Jump> {
    WTF_MAKE_TZONE_ALLOCATED(SaneStringGetByValSlowPathGenerator);
public:
    SaneStringGetByValSlowPathGenerator(
        const MacroAssembler::Jump& from, SpeculativeJIT* jit, JSValueRegs resultRegs, JITCompiler::LinkableConstant globalObject, GPRReg baseReg, GPRReg propertyReg)
        : JumpingSlowPathGenerator<MacroAssembler::Jump>(from, jit)
        , m_resultRegs(resultRegs)
        , m_globalObject(globalObject)
        , m_baseReg(baseReg)
        , m_propertyReg(propertyReg)
    {
        jit->silentSpillAllRegistersImpl(false, m_plans, extractResult(resultRegs));
    }
    
private:
    void generateInternal(SpeculativeJIT* jit) final
    {
        linkFrom(jit);
        
        MacroAssembler::Jump isNeg = jit->branch32(
            MacroAssembler::LessThan, m_propertyReg, MacroAssembler::TrustedImm32(0));
        
#if USE(JSVALUE64)
        jit->move(
            MacroAssembler::TrustedImm64(JSValue::encode(jsUndefined())), m_resultRegs.gpr());
#else
        jit->move(
            MacroAssembler::TrustedImm32(JSValue::UndefinedTag), m_resultRegs.tagGPR());
        jit->move(
            MacroAssembler::TrustedImm32(0), m_resultRegs.payloadGPR());
#endif
        jumpTo(jit);
        
        isNeg.link(jit);

        jit->callOperationWithSilentSpill(m_plans, operationGetByValStringInt, extractResult(m_resultRegs), m_globalObject, m_baseReg, m_propertyReg);
        
        jumpTo(jit);
    }

    JSValueRegs m_resultRegs;
    JITCompiler::LinkableConstant m_globalObject;
    GPRReg m_baseReg;
    GPRReg m_propertyReg;
    Vector<SilentRegisterSavePlan, 2> m_plans;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
