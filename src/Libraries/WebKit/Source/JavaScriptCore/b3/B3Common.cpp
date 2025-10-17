/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#include "B3Common.h"

#if ENABLE(B3_JIT)

#include "DFGCommon.h"
#include "FTLState.h"
#include "Options.h"

namespace JSC { namespace B3 {

const char* const tierName = "b3  ";

bool shouldDumpIR(Procedure& procedure, B3CompilationMode mode)
{
    if (procedure.shouldDumpIR())
        return true;

#if ENABLE(FTL_JIT)
    return FTL::verboseCompilationEnabled() || shouldDumpIRAtEachPhase(mode);
#else
    return shouldDumpIRAtEachPhase(mode);
#endif
}

bool shouldDumpIRAtEachPhase(B3CompilationMode mode)
{
    if (mode == B3Mode)
        return Options::dumpGraphAtEachPhase() || Options::dumpB3GraphAtEachPhase();
    return Options::dumpGraphAtEachPhase() || Options::dumpAirGraphAtEachPhase();
}

bool shouldValidateIR()
{
    return DFG::validationEnabled() || shouldValidateIRAtEachPhase();
}

bool shouldValidateIRAtEachPhase()
{
    return Options::validateGraphAtEachPhase();
}

bool shouldSaveIRBeforePhase()
{
    return Options::verboseValidationFailure();
}

GPRReg extendedOffsetAddrRegister()
{
    RELEASE_ASSERT(isARM64() || isRISCV64() || isARM_THUMB2());
#if CPU(ARM64) || CPU(RISCV64)
    return MacroAssembler::linkRegister;
#elif CPU(ARM)
    return MacroAssembler::dataTempRegister;
#elif CPU(X86_64)
    return GPRReg::InvalidGPRReg;
#else
#error Unhandled architecture.
#endif
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

