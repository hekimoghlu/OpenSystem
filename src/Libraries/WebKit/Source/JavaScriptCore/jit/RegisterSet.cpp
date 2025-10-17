/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#include "RegisterSet.h"

#if ENABLE(ASSEMBLER)

#include "FPRInfo.h"
#include "GPRInfo.h"
#include "MacroAssembler.h"
#include "RegisterAtOffsetList.h"

namespace JSC {

RegisterAtOffsetList* RegisterSetBuilder::vmCalleeSaveRegisterOffsets()
{
    static RegisterAtOffsetList* result;
    static std::once_flag calleeSavesFlag;
    std::call_once(calleeSavesFlag, [] () {
        result = new RegisterAtOffsetList(vmCalleeSaveRegisters(), RegisterAtOffsetList::ZeroBased);
#if USE(JSVALUE64)
        ASSERT(result->registerCount() == result->sizeOfAreaInBytes() / sizeof(CPURegister));
#endif
    });
    return result;
}

RegisterSet RegisterSetBuilder::stackRegisters()
{
    return RegisterSetBuilder(
        MacroAssembler::stackPointerRegister,
        MacroAssembler::framePointerRegister).buildAndValidate();
}

RegisterSet RegisterSetBuilder::reservedHardwareRegisters()
{
    RegisterSet result;

#define SET_IF_RESERVED(id, name, isReserved, isCalleeSaved)    \
    if (isReserved)                                             \
        result.add(RegisterNames::id, IgnoreVectors);
    FOR_EACH_GP_REGISTER(SET_IF_RESERVED)
    FOR_EACH_FP_REGISTER(SET_IF_RESERVED)
#undef SET_IF_RESERVED

    ASSERT(!result.numberOfSetFPRs());

    return result;
}

RegisterSet RegisterSetBuilder::runtimeTagRegisters()
{
#if USE(JSVALUE64)
    return RegisterSetBuilder(GPRInfo::numberTagRegister, GPRInfo::notCellMaskRegister).buildAndValidate();
#else
    return { };
#endif
}

RegisterSet RegisterSetBuilder::specialRegisters()
{
    return RegisterSetBuilder(
        RegisterSetBuilder::stackRegisters(),
        RegisterSetBuilder::reservedHardwareRegisters(),
        runtimeTagRegisters()).buildAndValidate();
}

RegisterSet RegisterSetBuilder::stubUnavailableRegisters()
{
    // FIXME: This is overly conservative. We could subtract out those callee-saves that we
    // actually saved.
    // https://bugs.webkit.org/show_bug.cgi?id=185686
    return RegisterSetBuilder(specialRegisters(), vmCalleeSaveRegisters()).buildAndValidate();
}

RegisterSet RegisterSetBuilder::macroClobberedGPRs()
{
#if CPU(X86_64)
    return RegisterSetBuilder(MacroAssembler::s_scratchRegister).buildAndValidate();
#elif CPU(ARM64) || CPU(RISCV64)
    return RegisterSetBuilder(MacroAssembler::dataTempRegister, MacroAssembler::memoryTempRegister).buildAndValidate();
#elif CPU(ARM_THUMB2)
    return RegisterSetBuilder(MacroAssembler::dataTempRegister, MacroAssembler::addressTempRegister).buildAndValidate();
#else
    return { };
#endif
}

RegisterSet RegisterSetBuilder::macroClobberedFPRs()
{
#if CPU(X86_64)
    RegisterSetBuilder builder;
    builder.add(MacroAssembler::fpTempRegister, IgnoreVectors);
    return builder.buildAndValidate();
#elif CPU(ARM64)
    RegisterSetBuilder builder;
    builder.add(MacroAssembler::fpTempRegister, IgnoreVectors);
    return builder.buildAndValidate();
#elif CPU(RISCV64)
    RegisterSetBuilder builder;
    builder.add(MacroAssembler::fpTempRegister, IgnoreVectors);
    builder.add(MacroAssembler::fpTempRegister2, IgnoreVectors);
    return builder.buildAndValidate();
#elif CPU(ARM_THUMB2)
    RegisterSetBuilder builder;
    builder.add(MacroAssembler::fpTempRegister, IgnoreVectors);
    return builder.buildAndValidate();
#else
    return { };
#endif
}

RegisterSet RegisterSetBuilder::calleeSaveRegisters()
{
    RegisterSet result;

#define SET_IF_CALLEESAVED(id, name, isReserved, isCalleeSaved)        \
    if (isCalleeSaved)                                                 \
        result.add(RegisterNames::id, IgnoreVectors);
    FOR_EACH_GP_REGISTER(SET_IF_CALLEESAVED)
#undef SET_IF_CALLEESAVED
#define SET_IF_CALLEESAVED(id, name, isReserved, isCalleeSaved)        \
    if (isCalleeSaved)                                                 \
        result.add(RegisterNames::id, Width64);
    FOR_EACH_FP_REGISTER(SET_IF_CALLEESAVED)
#undef SET_IF_CALLEESAVED

    return result;
}

RegisterSet RegisterSetBuilder::vmCalleeSaveRegisters()
{
    RegisterSet result;
#if CPU(X86_64)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
    result.add(GPRInfo::regCS2, IgnoreVectors);
    result.add(GPRInfo::regCS3, IgnoreVectors);
    result.add(GPRInfo::regCS4, IgnoreVectors);
#elif CPU(ARM64)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
    result.add(GPRInfo::regCS2, IgnoreVectors);
    result.add(GPRInfo::regCS3, IgnoreVectors);
    result.add(GPRInfo::regCS4, IgnoreVectors);
    result.add(GPRInfo::regCS5, IgnoreVectors);
    result.add(GPRInfo::regCS6, IgnoreVectors);
    result.add(GPRInfo::regCS7, IgnoreVectors);
    result.add(GPRInfo::regCS8, IgnoreVectors);
    result.add(GPRInfo::regCS9, IgnoreVectors);
    result.add(FPRInfo::fpRegCS0, Width64);
    result.add(FPRInfo::fpRegCS1, Width64);
    result.add(FPRInfo::fpRegCS2, Width64);
    result.add(FPRInfo::fpRegCS3, Width64);
    result.add(FPRInfo::fpRegCS4, Width64);
    result.add(FPRInfo::fpRegCS5, Width64);
    result.add(FPRInfo::fpRegCS6, Width64);
    result.add(FPRInfo::fpRegCS7, Width64);
#elif CPU(ARM_THUMB2)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
    result.add(FPRInfo::fpRegCS0, IgnoreVectors);
    result.add(FPRInfo::fpRegCS1, IgnoreVectors);
    result.add(FPRInfo::fpRegCS2, IgnoreVectors);
    result.add(FPRInfo::fpRegCS3, IgnoreVectors);
    result.add(FPRInfo::fpRegCS4, IgnoreVectors);
    result.add(FPRInfo::fpRegCS5, IgnoreVectors);
#elif CPU(RISCV64)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
    result.add(GPRInfo::regCS2, IgnoreVectors);
    result.add(GPRInfo::regCS3, IgnoreVectors);
    result.add(GPRInfo::regCS4, IgnoreVectors);
    result.add(GPRInfo::regCS5, IgnoreVectors);
    result.add(GPRInfo::regCS6, IgnoreVectors);
    result.add(GPRInfo::regCS7, IgnoreVectors);
    result.add(GPRInfo::regCS8, IgnoreVectors);
    result.add(GPRInfo::regCS9, IgnoreVectors);
    result.add(GPRInfo::regCS10, IgnoreVectors);
    result.add(FPRInfo::fpRegCS0, IgnoreVectors);
    result.add(FPRInfo::fpRegCS1, IgnoreVectors);
    result.add(FPRInfo::fpRegCS2, IgnoreVectors);
    result.add(FPRInfo::fpRegCS3, IgnoreVectors);
    result.add(FPRInfo::fpRegCS4, IgnoreVectors);
    result.add(FPRInfo::fpRegCS5, IgnoreVectors);
    result.add(FPRInfo::fpRegCS6, IgnoreVectors);
    result.add(FPRInfo::fpRegCS7, IgnoreVectors);
    result.add(FPRInfo::fpRegCS8, IgnoreVectors);
    result.add(FPRInfo::fpRegCS9, IgnoreVectors);
    result.add(FPRInfo::fpRegCS10, IgnoreVectors);
    result.add(FPRInfo::fpRegCS11, IgnoreVectors);
#endif
    return result;
}

RegisterSet RegisterSetBuilder::llintBaselineCalleeSaveRegisters()
{
    RegisterSet result;
#if CPU(X86_64)
    result.add(GPRInfo::regCS1, IgnoreVectors);
    static_assert(GPRInfo::regCS2 == GPRInfo::jitDataRegister);
    static_assert(GPRInfo::regCS3 == GPRInfo::numberTagRegister);
    static_assert(GPRInfo::regCS4 == GPRInfo::notCellMaskRegister);
    result.add(GPRInfo::regCS2, IgnoreVectors);
    result.add(GPRInfo::regCS3, IgnoreVectors);
    result.add(GPRInfo::regCS4, IgnoreVectors);
#elif CPU(ARM_THUMB2)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
#elif CPU(ARM64) || CPU(RISCV64)
    result.add(GPRInfo::regCS6, IgnoreVectors);
    static_assert(GPRInfo::regCS7 == GPRInfo::jitDataRegister);
    static_assert(GPRInfo::regCS8 == GPRInfo::numberTagRegister);
    static_assert(GPRInfo::regCS9 == GPRInfo::notCellMaskRegister);
    result.add(GPRInfo::regCS7, IgnoreVectors);
    result.add(GPRInfo::regCS8, IgnoreVectors);
    result.add(GPRInfo::regCS9, IgnoreVectors);
#else
    UNREACHABLE_FOR_PLATFORM();
#endif
    return result;
}

RegisterSet RegisterSetBuilder::dfgCalleeSaveRegisters()
{
    RegisterSet result;
#if CPU(X86_64)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
    static_assert(GPRInfo::regCS2 == GPRInfo::jitDataRegister);
    static_assert(GPRInfo::regCS3 == GPRInfo::numberTagRegister);
    static_assert(GPRInfo::regCS4 == GPRInfo::notCellMaskRegister);
    result.add(GPRInfo::regCS2, IgnoreVectors);
    result.add(GPRInfo::regCS3, IgnoreVectors);
    result.add(GPRInfo::regCS4, IgnoreVectors);
#elif CPU(ARM_THUMB2)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
#elif CPU(ARM64) || CPU(RISCV64)
    static_assert(GPRInfo::regCS7 == GPRInfo::jitDataRegister);
    static_assert(GPRInfo::regCS8 == GPRInfo::numberTagRegister);
    static_assert(GPRInfo::regCS9 == GPRInfo::notCellMaskRegister);
    result.add(GPRInfo::regCS7, IgnoreVectors);
    result.add(GPRInfo::regCS8, IgnoreVectors);
    result.add(GPRInfo::regCS9, IgnoreVectors);
#else
    UNREACHABLE_FOR_PLATFORM();
#endif
    return result;
}

RegisterSet RegisterSetBuilder::ftlCalleeSaveRegisters()
{
    RegisterSet result;
#if ENABLE(FTL_JIT)
#if CPU(X86_64)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
    static_assert(GPRInfo::regCS2 == GPRInfo::jitDataRegister);
    static_assert(GPRInfo::regCS3 == GPRInfo::numberTagRegister);
    static_assert(GPRInfo::regCS4 == GPRInfo::notCellMaskRegister);
    result.add(GPRInfo::regCS2, IgnoreVectors);
    result.add(GPRInfo::regCS3, IgnoreVectors);
    result.add(GPRInfo::regCS4, IgnoreVectors);
#elif CPU(ARM64)
    // B3 might save and use all ARM64 callee saves specified in the ABI.
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
    result.add(GPRInfo::regCS2, IgnoreVectors);
    result.add(GPRInfo::regCS3, IgnoreVectors);
    result.add(GPRInfo::regCS4, IgnoreVectors);
    result.add(GPRInfo::regCS5, IgnoreVectors);
    result.add(GPRInfo::regCS6, IgnoreVectors);
    static_assert(GPRInfo::regCS7 == GPRInfo::jitDataRegister);
    static_assert(GPRInfo::regCS8 == GPRInfo::numberTagRegister);
    static_assert(GPRInfo::regCS9 == GPRInfo::notCellMaskRegister);
    result.add(GPRInfo::regCS7, IgnoreVectors);
    result.add(GPRInfo::regCS8, IgnoreVectors);
    result.add(GPRInfo::regCS9, IgnoreVectors);
    result.add(FPRInfo::fpRegCS0, Width64);
    result.add(FPRInfo::fpRegCS1, Width64);
    result.add(FPRInfo::fpRegCS2, Width64);
    result.add(FPRInfo::fpRegCS3, Width64);
    result.add(FPRInfo::fpRegCS4, Width64);
    result.add(FPRInfo::fpRegCS5, Width64);
    result.add(FPRInfo::fpRegCS6, Width64);
    result.add(FPRInfo::fpRegCS7, Width64);
#elif CPU(RISCV64)
    result.add(GPRInfo::regCS0, IgnoreVectors);
    result.add(GPRInfo::regCS1, IgnoreVectors);
    result.add(GPRInfo::regCS2, IgnoreVectors);
    result.add(GPRInfo::regCS3, IgnoreVectors);
    result.add(GPRInfo::regCS4, IgnoreVectors);
    result.add(GPRInfo::regCS5, IgnoreVectors);
    result.add(GPRInfo::regCS6, IgnoreVectors);
    static_assert(GPRInfo::regCS7 == GPRInfo::jitDataRegister);
    static_assert(GPRInfo::regCS8 == GPRInfo::numberTagRegister);
    static_assert(GPRInfo::regCS9 == GPRInfo::notCellMaskRegister);
    result.add(GPRInfo::regCS7, IgnoreVectors);
    result.add(GPRInfo::regCS8, IgnoreVectors);
    result.add(GPRInfo::regCS9, IgnoreVectors);
    result.add(GPRInfo::regCS10, IgnoreVectors);
    result.add(FPRInfo::fpRegCS0, IgnoreVectors);
    result.add(FPRInfo::fpRegCS1, IgnoreVectors);
    result.add(FPRInfo::fpRegCS2, IgnoreVectors);
    result.add(FPRInfo::fpRegCS3, IgnoreVectors);
    result.add(FPRInfo::fpRegCS4, IgnoreVectors);
    result.add(FPRInfo::fpRegCS5, IgnoreVectors);
    result.add(FPRInfo::fpRegCS6, IgnoreVectors);
    result.add(FPRInfo::fpRegCS7, IgnoreVectors);
    result.add(FPRInfo::fpRegCS8, IgnoreVectors);
    result.add(FPRInfo::fpRegCS9, IgnoreVectors);
    result.add(FPRInfo::fpRegCS10, IgnoreVectors);
    result.add(FPRInfo::fpRegCS11, IgnoreVectors);
#else
    UNREACHABLE_FOR_PLATFORM();
#endif
#endif
    return result;
}

RegisterSet RegisterSetBuilder::argumentGPRs()
{
    RegisterSet result;
#if NUMBER_OF_ARGUMENT_REGISTERS
    for (unsigned i = 0; i < GPRInfo::numberOfArgumentRegisters; i++)
        result.add(GPRInfo::toArgumentRegister(i), IgnoreVectors);
#endif
    return result;
}

RegisterSet RegisterSetBuilder::argumentFPRs()
{
    RegisterSet result;
#if NUMBER_OF_ARGUMENT_REGISTERS
    for (unsigned i = 0; i < FPRInfo::numberOfArgumentRegisters; i++)
        result.add(FPRInfo::toArgumentRegister(i), IgnoreVectors);
#endif
    return result;
}

RegisterSetBuilder RegisterSetBuilder::registersToSaveForJSCall(RegisterSetBuilder liveRegisters)
{
    RegisterSetBuilder result = liveRegisters;
    result.exclude(RegisterSetBuilder::vmCalleeSaveRegisters());
    result.exclude(RegisterSetBuilder::stackRegisters());
    result.exclude(RegisterSetBuilder::reservedHardwareRegisters());
    return result;
}

RegisterSetBuilder RegisterSetBuilder::registersToSaveForCCall(RegisterSetBuilder liveRegisters)
{
    RegisterSetBuilder result = liveRegisters;
    result.exclude(RegisterSetBuilder::calleeSaveRegisters());
    result.exclude(RegisterSetBuilder::stackRegisters());
    result.exclude(RegisterSetBuilder::reservedHardwareRegisters());
    return result;
}

RegisterSet RegisterSetBuilder::allGPRs()
{
    RegisterSet result;
    for (MacroAssembler::RegisterID reg = MacroAssembler::firstRegister(); reg <= MacroAssembler::lastRegister(); reg = static_cast<MacroAssembler::RegisterID>(reg + 1))
        result.add(reg, IgnoreVectors);
    return result;
}

RegisterSet RegisterSetBuilder::allFPRs()
{
    RegisterSet result;
    for (MacroAssembler::FPRegisterID reg = MacroAssembler::firstFPRegister(); reg <= MacroAssembler::lastFPRegister(); reg = static_cast<MacroAssembler::FPRegisterID>(reg + 1))
        result.add(reg, conservativeWidth(reg));
    return result;
}

RegisterSet RegisterSetBuilder::allRegisters()
{
    RegisterSet result;
    result.merge(allGPRs());
    result.merge(allFPRs());
    return result;
}

RegisterSet RegisterSetBuilder::allScalarRegisters()
{
    RegisterSet result;
    result.merge(allGPRs());
    result.merge(allFPRs());
    result.m_upperBits.clearAll();
    return result;
}

#if ENABLE(WEBASSEMBLY)
RegisterSet RegisterSetBuilder::wasmPinnedRegisters()
{
    RegisterSet result;
    if constexpr (GPRInfo::wasmBaseMemoryPointer != InvalidGPRReg)
        result.add(GPRInfo::wasmBaseMemoryPointer, IgnoreVectors);
    if constexpr (GPRInfo::wasmContextInstancePointer != InvalidGPRReg)
        result.add(GPRInfo::wasmContextInstancePointer, IgnoreVectors);
    if constexpr (GPRInfo::wasmBoundsCheckingSizeRegister != InvalidGPRReg)
        result.add(GPRInfo::wasmBoundsCheckingSizeRegister, IgnoreVectors);
    return result;
}
#endif

} // namespace JSC

#endif // ENABLE(ASSEMBLER)

