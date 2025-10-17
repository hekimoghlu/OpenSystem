/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#include "AirCCallSpecial.h"

#if ENABLE(B3_JIT)

#include "CCallHelpers.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace B3 { namespace Air {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CCallSpecial);

CCallSpecial::CCallSpecial(bool isSIMDContext)
    : m_isSIMDContext(isSIMDContext)
{
    m_clobberedRegs = RegisterSetBuilder::registersToSaveForCCall(m_isSIMDContext ? RegisterSetBuilder::allRegisters() : RegisterSetBuilder::allScalarRegisters());
    m_clobberedRegs.remove(GPRInfo::returnValueGPR);
    m_clobberedRegs.remove(GPRInfo::returnValueGPR2);
    m_clobberedRegs.remove(FPRInfo::returnValueFPR);
}

CCallSpecial::~CCallSpecial() = default;

void CCallSpecial::forEachArg(Inst& inst, const ScopedLambda<Inst::EachArgCallback>& callback)
{
    for (unsigned i = 0; i < numCalleeArgs; ++i)
        callback(inst.args[calleeArgOffset + i], Arg::Use, GP, pointerWidth());
    for (unsigned i = 0; i < numReturnGPArgs; ++i)
        callback(inst.args[returnGPArgOffset + i], Arg::Def, GP, pointerWidth());
    for (unsigned i = 0; i < numReturnFPArgs; ++i)
        callback(inst.args[returnFPArgOffset + i], Arg::Def, FP, m_isSIMDContext ? conservativeWidth(FP) : conservativeWidthWithoutVectors(FP));
    
    for (unsigned i = argArgOffset; i < inst.args.size(); ++i) {
        // For the type, we can just query the arg's bank. The arg will have a bank, because we
        // require these args to be argument registers.
        Bank bank = inst.args[i].bank();
        callback(inst.args[i], Arg::Use, bank, m_isSIMDContext ? conservativeWidth(bank) : conservativeWidthWithoutVectors(bank));
    }
}

bool CCallSpecial::isValid(Inst& inst)
{
    if (inst.args.size() < argArgOffset)
        return false;

    for (unsigned i = 0; i < numCalleeArgs; ++i) {
        Arg& arg = inst.args[i + calleeArgOffset];
        if (!arg.isGP())
            return false;
        switch (arg.kind()) {
        case Arg::Imm:
            if (is32Bit())
                break;
            return false;
        case Arg::BigImm:
            if (is64Bit())
                break;
            return false;
        case Arg::Tmp:
        case Arg::Addr:
        case Arg::ExtendedOffsetAddr:
        case Arg::Stack:
        case Arg::CallArg:
            break;
        default:
            return false;
        }
    }

    // Return args need to be exact.
    if (inst.args[returnGPArgOffset + 0] != Tmp(GPRInfo::returnValueGPR))
        return false;
    if (inst.args[returnGPArgOffset + 1] != Tmp(GPRInfo::returnValueGPR2))
        return false;
    if (inst.args[returnFPArgOffset + 0] != Tmp(FPRInfo::returnValueFPR))
        return false;

    for (unsigned i = argArgOffset; i < inst.args.size(); ++i) {
        if (!inst.args[i].isReg())
            return false;

        if (inst.args[i] == Tmp(scratchRegister))
            return false;
    }
    return true;
}

bool CCallSpecial::admitsStack(Inst&, unsigned argIndex)
{
    // The callee can be on the stack unless targeting ARM64, where we can't later properly
    // handle an Addr callee argument in generate() due to disallowed scratch register usage.
    if (argIndex == calleeArgOffset)
        return !isARM64();
    
    return false;
}

bool CCallSpecial::admitsExtendedOffsetAddr(Inst& inst, unsigned argIndex)
{
    return admitsStack(inst, argIndex);
}

void CCallSpecial::reportUsedRegisters(Inst&, const RegisterSetBuilder&)
{
}

CCallHelpers::Jump CCallSpecial::generate(Inst& inst, CCallHelpers& jit, GenerationContext&)
{
    switch (inst.args[calleeArgOffset].kind()) {
    case Arg::Imm:
    case Arg::BigImm:
        jit.move(inst.args[calleeArgOffset].asTrustedImmPtr(), scratchRegister);
        jit.call(scratchRegister, OperationPtrTag);
        break;
    case Arg::Tmp:
        jit.call(inst.args[calleeArgOffset].gpr(), OperationPtrTag);
        break;
    case Arg::Addr:
    case Arg::ExtendedOffsetAddr:
        jit.call(inst.args[calleeArgOffset].asAddress(), OperationPtrTag);
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        break;
    }
    return CCallHelpers::Jump();
}

RegisterSetBuilder CCallSpecial::extraEarlyClobberedRegs(Inst&)
{
    return { };
}

RegisterSetBuilder CCallSpecial::extraClobberedRegs(Inst&)
{
    return m_clobberedRegs;
}

void CCallSpecial::dumpImpl(PrintStream& out) const
{
    out.print("CCall");
}

void CCallSpecial::deepDumpImpl(PrintStream& out) const
{
    out.print("function call that uses the C calling convention.");
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
