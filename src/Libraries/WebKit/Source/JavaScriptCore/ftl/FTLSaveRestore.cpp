/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#include "FTLSaveRestore.h"

#if ENABLE(FTL_JIT)

#include "AssemblyHelpersSpoolers.h"
#include "FPRInfo.h"
#include "GPRInfo.h"
#include "Reg.h"
#include "RegisterSet.h"

namespace JSC { namespace FTL {

static size_t bytesForGPRs()
{
    return MacroAssembler::numberOfRegisters() * sizeof(int64_t);
}

static size_t bytesForFPRs()
{
    // FIXME: It might be worthwhile saving the full state of the FP registers, at some point.
    // Right now we don't need this since we only do the save/restore just prior to OSR exit, and
    // OSR exit will be guaranteed to only need the double portion of the FP registers.
    return MacroAssembler::numberOfFPRegisters() * sizeof(double);
}

size_t requiredScratchMemorySizeInBytes()
{
    return bytesForGPRs() + bytesForFPRs();
}

size_t offsetOfGPR(GPRReg reg)
{
    return MacroAssembler::registerIndex(reg) * sizeof(int64_t);
}

size_t offsetOfFPR(FPRReg reg)
{
    return bytesForGPRs() + MacroAssembler::fpRegisterIndex(reg) * sizeof(double);
}

size_t offsetOfReg(Reg reg)
{
    if (reg.isGPR())
        return offsetOfGPR(reg.gpr());
    return offsetOfFPR(reg.fpr());
}

namespace {

struct Regs {
    Regs()
    {
        special = RegisterSetBuilder::stackRegisters();
        special.merge(RegisterSetBuilder::reservedHardwareRegisters());

        first = MacroAssembler::firstRegister();
        while (special.contains(first, IgnoreVectors))
            first = MacroAssembler::nextRegister(first);
    }

    GPRReg nextRegister(GPRReg current)
    {
        auto next = MacroAssembler::nextRegister(current);
        for (; next <= MacroAssembler::lastRegister(); next = MacroAssembler::nextRegister(next)) {
            if (!special.contains(next, IgnoreVectors))
                break;
        }
        return next;
    }

    RegisterSet special;
    GPRReg first;
};

} // anonymous namespace

void saveAllRegisters(AssemblyHelpers& jit, char* scratchMemory)
{
    Regs regs;
    
    // Get the first register out of the way, so that we can use it as a pointer.
    GPRReg baseGPR = regs.first;
#if CPU(ARM64)
    GPRReg nextGPR = regs.nextRegister(baseGPR);
    GPRReg firstToSaveGPR = regs.nextRegister(nextGPR);
    ASSERT(baseGPR == ARM64Registers::x0);
    ASSERT(nextGPR == ARM64Registers::x1);
#else
    GPRReg firstToSaveGPR = regs.nextRegister(baseGPR);
#endif
    jit.poke64(baseGPR, 0);
    jit.move(MacroAssembler::TrustedImmPtr(scratchMemory), baseGPR);

    AssemblyHelpers::StoreRegSpooler spooler(jit, baseGPR);

    // Get all of the other GPRs out of the way.
    for (MacroAssembler::RegisterID reg = firstToSaveGPR; reg <= MacroAssembler::lastRegister(); reg = MacroAssembler::nextRegister(reg)) {
        if (regs.special.contains(reg, IgnoreVectors))
            continue;
        spooler.storeGPR({ reg, static_cast<ptrdiff_t>(offsetOfGPR(reg)), conservativeWidthWithoutVectors(reg) });
    }
    spooler.finalizeGPR();
    
    // Restore the first register into the second one and save it.
    jit.peek64(firstToSaveGPR, 0);
#if CPU(ARM64)
    jit.storePair64(firstToSaveGPR, nextGPR, baseGPR, AssemblyHelpers::TrustedImm32(offsetOfGPR(baseGPR)));
#else
    jit.store64(firstToSaveGPR, MacroAssembler::Address(baseGPR, offsetOfGPR(baseGPR)));
#endif
    
    // Finally save all FPR's.
    for (MacroAssembler::FPRegisterID reg = MacroAssembler::firstFPRegister(); reg <= MacroAssembler::lastFPRegister(); reg = MacroAssembler::nextFPRegister(reg)) {
        if (regs.special.contains(reg, IgnoreVectors))
            continue;
        spooler.storeFPR({ reg, static_cast<ptrdiff_t>(offsetOfFPR(reg)), conservativeWidthWithoutVectors(reg) });
    }
    spooler.finalizeFPR();
}

void restoreAllRegisters(AssemblyHelpers& jit, char* scratchMemory)
{
    Regs regs;
    
    // Give ourselves a pointer to the scratch memory.
    GPRReg baseGPR = regs.first;
    jit.move(MacroAssembler::TrustedImmPtr(scratchMemory), baseGPR);
    
    AssemblyHelpers::LoadRegSpooler spooler(jit, baseGPR);

    // Restore all FPR's.
    for (MacroAssembler::FPRegisterID reg = MacroAssembler::firstFPRegister(); reg <= MacroAssembler::lastFPRegister(); reg = MacroAssembler::nextFPRegister(reg)) {
        if (regs.special.contains(reg, IgnoreVectors))
            continue;
        spooler.loadFPR({ reg, static_cast<ptrdiff_t>(offsetOfFPR(reg)), conservativeWidthWithoutVectors(reg) });
    }
    spooler.finalizeFPR();
    
#if CPU(ARM64)
    GPRReg nextGPR = regs.nextRegister(baseGPR);
    GPRReg firstToRestoreGPR = regs.nextRegister(nextGPR);
    ASSERT(baseGPR == ARM64Registers::x0);
    ASSERT(nextGPR == ARM64Registers::x1);
#else
    GPRReg firstToRestoreGPR = regs.nextRegister(baseGPR);
#endif
    for (MacroAssembler::RegisterID reg = firstToRestoreGPR; reg <= MacroAssembler::lastRegister(); reg = MacroAssembler::nextRegister(reg)) {
        if (regs.special.contains(reg, IgnoreVectors))
            continue;
        spooler.loadGPR({ reg, static_cast<ptrdiff_t>(offsetOfGPR(reg)), conservativeWidthWithoutVectors(reg) });
    }
    spooler.finalizeGPR();

#if CPU(ARM64)
    jit.loadPair64(baseGPR, AssemblyHelpers::TrustedImm32(offsetOfGPR(baseGPR)), baseGPR, nextGPR);
#else
    jit.load64(MacroAssembler::Address(baseGPR, offsetOfGPR(baseGPR)), baseGPR);
#endif
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

