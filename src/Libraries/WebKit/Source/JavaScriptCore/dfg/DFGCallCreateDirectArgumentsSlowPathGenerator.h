/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include "DirectArguments.h"

namespace JSC { namespace DFG {

// This calls operationCreateDirectArguments but then restores the value of lengthGPR.
class CallCreateDirectArgumentsSlowPathGenerator final : public JumpingSlowPathGenerator<MacroAssembler::JumpList> {
    WTF_MAKE_TZONE_ALLOCATED(CallCreateDirectArgumentsSlowPathGenerator);
public:
    CallCreateDirectArgumentsSlowPathGenerator(
        MacroAssembler::JumpList from, SpeculativeJIT* jit, GPRReg resultGPR, RegisteredStructure structure,
        GPRReg lengthGPR, unsigned minCapacity)
        : JumpingSlowPathGenerator<MacroAssembler::JumpList>(from, jit)
        , m_resultGPR(resultGPR)
        , m_structure(structure)
        , m_lengthGPR(lengthGPR)
        , m_minCapacity(minCapacity)
    {
        jit->silentSpillAllRegistersImpl(false, m_plans, resultGPR);
    }

private:
    void generateInternal(SpeculativeJIT* jit) final
    {
        linkFrom(jit);
        jit->callOperationWithSilentSpill(m_plans.span(), operationCreateDirectArguments, m_resultGPR, SpeculativeJIT::TrustedImmPtr(&jit->vm()), m_structure, m_lengthGPR, m_minCapacity);
        jit->loadPtr(
            MacroAssembler::Address(m_resultGPR, DirectArguments::offsetOfLength()), m_lengthGPR);
        jumpTo(jit);
    }

    GPRReg m_resultGPR;
    RegisteredStructure m_structure;
    GPRReg m_lengthGPR;
    unsigned m_minCapacity;
    Vector<SilentRegisterSavePlan, 2> m_plans;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
