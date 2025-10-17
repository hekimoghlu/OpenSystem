/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#include "AirPrintSpecial.h"

#if ENABLE(B3_JIT)

#include "CCallHelpers.h"
#include "MacroAssemblerPrinter.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace B3 { namespace Air {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PrintSpecial);

PrintSpecial::PrintSpecial(Printer::PrintRecordList* list)
    : m_printRecordList(list)
{
}

PrintSpecial::~PrintSpecial() = default;

void PrintSpecial::forEachArg(Inst&, const ScopedLambda<Inst::EachArgCallback>&)
{
}

bool PrintSpecial::isValid(Inst&)
{
    return true;
}

bool PrintSpecial::admitsStack(Inst&, unsigned)
{
    return false;
}

bool PrintSpecial::admitsExtendedOffsetAddr(Inst&, unsigned)
{
    return false;
}

void PrintSpecial::reportUsedRegisters(Inst&, const RegisterSetBuilder&)
{
}

MacroAssembler::Jump PrintSpecial::generate(Inst& inst, CCallHelpers& jit, GenerationContext&)
{
    size_t currentArg = 1; // Skip the PrintSpecial arg.
    for (auto& term : *m_printRecordList) {
        if (term.printer == Printer::printAirArg) {
            const Arg& arg = inst.args[currentArg++];
            switch (arg.kind()) {
            case Arg::Tmp:
                term = Printer::Printer<MacroAssembler::RegisterID>(arg.gpr());
                break;
            case Arg::Addr:
            case Arg::ExtendedOffsetAddr:
                term = Printer::Printer<MacroAssembler::Address>(arg.asAddress());
                break;
            default:
                RELEASE_ASSERT_NOT_REACHED();
                break;
            }
        }
    }
    jit.print(m_printRecordList);
    return CCallHelpers::Jump();
}

RegisterSetBuilder PrintSpecial::extraEarlyClobberedRegs(Inst&)
{
    return { };
}

RegisterSetBuilder PrintSpecial::extraClobberedRegs(Inst&)
{
    return { };
}

void PrintSpecial::dumpImpl(PrintStream& out) const
{
    out.print("Print");
}

void PrintSpecial::deepDumpImpl(PrintStream& out) const
{
    out.print("print for debugging logging.");
}

} } // namespace B3::Air

namespace Printer {

NO_RETURN void printAirArg(PrintStream&, Context&)
{
    // This function is only a placeholder to let PrintSpecial::generate() know that
    // the Printer needs to be replaced with one for a register, constant, etc. Hence,
    // this function should never be called.
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace Printer

} // namespace JSC

#endif // ENABLE(B3_JIT)
