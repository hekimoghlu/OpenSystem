/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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

#include "DFGSlowPathGenerator.h"
#include "DFGSpeculativeJIT.h"
#include <wtf/Vector.h>

namespace JSC { namespace DFG {

class CallArrayAllocatorSlowPathGenerator final : public JumpingSlowPathGenerator<MacroAssembler::JumpList> {
    WTF_MAKE_TZONE_ALLOCATED(CallArrayAllocatorSlowPathGenerator);
public:
    CallArrayAllocatorSlowPathGenerator(
        MacroAssembler::JumpList from, SpeculativeJIT* jit, P_JITOperation_VmStZB function,
        GPRReg resultGPR, GPRReg storageGPR, RegisteredStructure structure, size_t size)
        : JumpingSlowPathGenerator<MacroAssembler::JumpList>(from, jit)
        , m_function(function)
        , m_resultGPR(resultGPR)
        , m_storageGPR(storageGPR)
        , m_size(size)
        , m_structure(structure)
    {
        ASSERT(size < static_cast<size_t>(std::numeric_limits<int32_t>::max()));
        jit->silentSpillAllRegistersImpl(false, m_plans, resultGPR);
    }

private:
    void generateInternal(SpeculativeJIT* jit) final
    {
        linkFrom(jit);
        jit->callOperationWithSilentSpill(m_plans.span(), m_function, m_resultGPR, SpeculativeJIT::TrustedImmPtr(&jit->vm()), m_structure, m_size, m_storageGPR);
        jit->loadPtr(MacroAssembler::Address(m_resultGPR, JSObject::butterflyOffset()), m_storageGPR);
        jumpTo(jit);
    }

    P_JITOperation_VmStZB m_function;
    GPRReg m_resultGPR;
    GPRReg m_storageGPR;
    int m_size;
    RegisteredStructure m_structure;
    Vector<SilentRegisterSavePlan, 2> m_plans;
};

class CallArrayAllocatorWithVariableSizeSlowPathGenerator final : public JumpingSlowPathGenerator<MacroAssembler::JumpList> {
    WTF_MAKE_TZONE_ALLOCATED(CallArrayAllocatorWithVariableSizeSlowPathGenerator);
public:
    CallArrayAllocatorWithVariableSizeSlowPathGenerator(
        MacroAssembler::JumpList from, SpeculativeJIT* jit, P_JITOperation_GStZB function,
        GPRReg resultGPR, JITCompiler::LinkableConstant globalObject, RegisteredStructure contiguousStructure, RegisteredStructure arrayStorageStructure, GPRReg sizeGPR, GPRReg storageGPR)
        : JumpingSlowPathGenerator<MacroAssembler::JumpList>(from, jit)
        , m_function(function)
        , m_contiguousStructure(contiguousStructure)
        , m_arrayStorageOrContiguousStructure(arrayStorageStructure)
        , m_resultGPR(resultGPR)
        , m_globalObject(globalObject)
        , m_sizeGPR(sizeGPR)
        , m_storageGPR(storageGPR)
    {
        jit->silentSpillAllRegistersImpl(false, m_plans, resultGPR);
    }

private:
    void generateInternal(SpeculativeJIT* jit) final
    {
        linkFrom(jit);
        jit->silentSpill(m_plans);
        GPRReg scratchGPR = AssemblyHelpers::selectScratchGPR(m_sizeGPR, m_storageGPR);
        if (m_contiguousStructure.get() != m_arrayStorageOrContiguousStructure.get()) {
            MacroAssembler::Jump bigLength = jit->branch32(MacroAssembler::AboveOrEqual, m_sizeGPR, MacroAssembler::TrustedImm32(MIN_ARRAY_STORAGE_CONSTRUCTION_LENGTH));
            jit->move(SpeculativeJIT::TrustedImmPtr(m_contiguousStructure), scratchGPR);
            MacroAssembler::Jump done = jit->jump();
            bigLength.link(jit);
            jit->move(SpeculativeJIT::TrustedImmPtr(m_arrayStorageOrContiguousStructure), scratchGPR);
            done.link(jit);
        } else
            jit->move(SpeculativeJIT::TrustedImmPtr(m_contiguousStructure), scratchGPR);
        jit->setupArguments<decltype(m_function)>(m_globalObject, scratchGPR, m_sizeGPR, m_storageGPR);
        jit->appendCall(m_function);
        std::optional<GPRReg> exception = jit->tryHandleOrGetExceptionUnderSilentSpill<decltype(m_function)>(m_plans, m_resultGPR);
        jit->setupResults(m_resultGPR);
        jit->silentFill(m_plans);

        if (exception)
            jit->exceptionCheck(*exception);

        jumpTo(jit);
    }

    P_JITOperation_GStZB m_function;
    RegisteredStructure m_contiguousStructure;
    RegisteredStructure m_arrayStorageOrContiguousStructure;
    GPRReg m_resultGPR;
    JITCompiler::LinkableConstant m_globalObject;
    GPRReg m_sizeGPR;
    GPRReg m_storageGPR;
    Vector<SilentRegisterSavePlan, 2> m_plans;
};

class CallArrayAllocatorWithVariableStructureVariableSizeSlowPathGenerator final : public JumpingSlowPathGenerator<MacroAssembler::JumpList> {
    WTF_MAKE_TZONE_ALLOCATED(CallArrayAllocatorWithVariableStructureVariableSizeSlowPathGenerator);
public:
    CallArrayAllocatorWithVariableStructureVariableSizeSlowPathGenerator(
        MacroAssembler::JumpList from, SpeculativeJIT* jit, P_JITOperation_GStZB function,
        GPRReg resultGPR, JITCompiler::LinkableConstant globalObject, GPRReg structureGPR, GPRReg sizeGPR, GPRReg storageGPR)
        : JumpingSlowPathGenerator<MacroAssembler::JumpList>(from, jit)
        , m_function(function)
        , m_resultGPR(resultGPR)
        , m_globalObject(globalObject)
        , m_structureGPR(structureGPR)
        , m_sizeGPR(sizeGPR)
        , m_storageGPR(storageGPR)
    {
        jit->silentSpillAllRegistersImpl(false, m_plans, resultGPR);
    }

private:
    void generateInternal(SpeculativeJIT* jit) final
    {
        linkFrom(jit);
        jit->callOperationWithSilentSpill(m_plans, m_function, m_resultGPR, m_globalObject, m_structureGPR, m_sizeGPR, m_storageGPR);
        jumpTo(jit);
    }

    P_JITOperation_GStZB m_function;
    GPRReg m_resultGPR;
    JITCompiler::LinkableConstant m_globalObject;
    GPRReg m_structureGPR;
    GPRReg m_sizeGPR;
    GPRReg m_storageGPR;
    Vector<SilentRegisterSavePlan, 2> m_plans;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
