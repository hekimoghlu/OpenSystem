/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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
#include "WasmCallingConvention.h"

#if ENABLE(WEBASSEMBLY)

#include <wtf/NeverDestroyed.h>

namespace JSC::Wasm {

const JSCallingConvention& jsCallingConvention()
{
    static LazyNeverDestroyed<JSCallingConvention> staticJSCallingConvention;
    static std::once_flag staticJSCCallingConventionFlag;
    std::call_once(staticJSCCallingConventionFlag, [] () {
        staticJSCallingConvention.construct(Vector<JSValueRegs>(), Vector<FPRReg>(), RegisterSetBuilder::calleeSaveRegisters());
    });

    return staticJSCallingConvention;
}

const WasmCallingConvention& wasmCallingConvention()
{
    static LazyNeverDestroyed<WasmCallingConvention> staticWasmCallingConvention;
    static std::once_flag staticWasmCallingConventionFlag;
    std::call_once(staticWasmCallingConventionFlag, [] () {
#if USE(JSVALUE64) // One value per GPR
        constexpr unsigned numberOfArgumentJSRs = GPRInfo::numberOfArgumentRegisters;
#elif USE(JSVALUE32_64) // One value per consecutive GPR pair
        constexpr unsigned numberOfArgumentJSRs = GPRInfo::numberOfArgumentRegisters / 2;
#endif
        Vector<JSValueRegs> jsrArgumentRegisters(numberOfArgumentJSRs);
        for (unsigned i = 0; i < numberOfArgumentJSRs; ++i) {
#if USE(JSVALUE64)
            jsrArgumentRegisters[i] = JSValueRegs { GPRInfo::toArgumentRegister(i) };
#elif USE(JSVALUE32_64)
            jsrArgumentRegisters[i] = JSValueRegs { GPRInfo::toArgumentRegister(2 * i + 1), GPRInfo::toArgumentRegister(2 * i) };
#endif
        }

        Vector<FPRReg> fprArgumentRegisters(FPRInfo::numberOfArgumentRegisters);
        for (unsigned i = 0; i < FPRInfo::numberOfArgumentRegisters; ++i)
            fprArgumentRegisters[i] = FPRInfo::toArgumentRegister(i);

        RegisterSetBuilder scratch = RegisterSetBuilder::allGPRs();
        scratch.exclude(RegisterSetBuilder::vmCalleeSaveRegisters().includeWholeRegisterWidth());
        scratch.exclude(RegisterSetBuilder::macroClobberedGPRs());
        scratch.exclude(RegisterSetBuilder::reservedHardwareRegisters());
        scratch.exclude(RegisterSetBuilder::stackRegisters());
        for (JSValueRegs jsr : jsrArgumentRegisters) {
            scratch.remove(jsr.payloadGPR());
#if USE(JSVALUE32_64)
            scratch.remove(jsr.tagGPR());
#endif
        }

        Vector<GPRReg> scratchGPRs;
        for (Reg reg : scratch.buildAndValidate())
            scratchGPRs.append(reg.gpr());

        // Need at least one JSValue and an additional GPR
#if USE(JSVALUE64)
        RELEASE_ASSERT(scratchGPRs.size() >= 2);
#elif USE(JSVALUE32_64)
        RELEASE_ASSERT(scratchGPRs.size() >= 3);
#endif

        staticWasmCallingConvention.construct(WTFMove(jsrArgumentRegisters), WTFMove(fprArgumentRegisters), WTFMove(scratchGPRs), RegisterSetBuilder::calleeSaveRegisters());
    });

    return staticWasmCallingConvention;
}

#if CPU(ARM_THUMB2)

const CCallingConventionArmThumb2& cCallingConventionArmThumb2()
{
    static LazyNeverDestroyed<CCallingConventionArmThumb2> staticCCallingConventionArmThumb2;
    static std::once_flag staticCCallingConventionArmThumb2Flag;
    std::call_once(staticCCallingConventionArmThumb2Flag, [] () {
        constexpr unsigned numberOfArgumentGPRs = GPRInfo::numberOfArgumentRegisters;
        Vector<GPRReg> gprArgumentRegisters(numberOfArgumentGPRs);
        for (unsigned i = 0; i < numberOfArgumentGPRs; ++i)
            gprArgumentRegisters[i] = GPRInfo::toArgumentRegister(i);

        Vector<FPRReg> fprArgumentRegisters(FPRInfo::numberOfArgumentRegisters);
        for (unsigned i = 0; i < FPRInfo::numberOfArgumentRegisters; ++i)
            fprArgumentRegisters[i] = FPRInfo::toArgumentRegister(i);

        RegisterSetBuilder scratch = RegisterSetBuilder::allGPRs();
        scratch.exclude(RegisterSetBuilder::vmCalleeSaveRegisters().includeWholeRegisterWidth());
        scratch.exclude(RegisterSetBuilder::macroClobberedGPRs());
        scratch.exclude(RegisterSetBuilder::reservedHardwareRegisters());
        scratch.exclude(RegisterSetBuilder::stackRegisters());
        for (GPRReg gpr : gprArgumentRegisters)
            scratch.remove(gpr);

        Vector<GPRReg> scratchGPRs;
        for (Reg reg : scratch.buildAndValidate())
            scratchGPRs.append(reg.gpr());

        // Need at least one JSValue and an additional GPR
        RELEASE_ASSERT(scratchGPRs.size() >= 3);

        staticCCallingConventionArmThumb2.construct(WTFMove(gprArgumentRegisters), WTFMove(fprArgumentRegisters), WTFMove(scratchGPRs), RegisterSetBuilder::calleeSaveRegisters());
    });

    return staticCCallingConventionArmThumb2;
}

#endif

} // namespace JSC::Wasm

#endif // ENABLE(B3_JIT)
