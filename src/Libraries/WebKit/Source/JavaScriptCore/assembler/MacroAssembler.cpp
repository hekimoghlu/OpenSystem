/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
#include "MacroAssembler.h"

#if ENABLE(ASSEMBLER)

#include "JSCPtrTag.h"
#include "Options.h"
#include "ProbeContext.h"
#include <wtf/PrintStream.h>
#include <wtf/ScopedLambda.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

const double MacroAssembler::twoToThe32 = (double)0x100000000ull;

void MacroAssembler::jitAssert(const ScopedLambda<Jump(void)>& functor)
{
    if (Options::useJITDebugAssertions()) {
        Jump passed = functor();
        breakpoint();
        passed.link(this);
    }
}

static void SYSV_ABI stdFunctionCallback(Probe::Context& context)
{
    auto func = context.arg<const Function<void(Probe::Context&)>*>();
    (*func)(context);
}
    
void MacroAssembler::probeDebug(Function<void(Probe::Context&)> func)
{
    probe(tagCFunction<JITProbePtrTag>(stdFunctionCallback), new Function<void(Probe::Context&)>(WTFMove(func)));
}

void MacroAssembler::probeDebugSIMD(Function<void(Probe::Context&)> func)
{
    probe(tagCFunction<JITProbePtrTag>(stdFunctionCallback), new Function<void(Probe::Context&)>(WTFMove(func)), Probe::SavedFPWidth::SaveVectors);
}

} // namespace JSC

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, MacroAssembler::RelationalCondition cond)
{
    switch (cond) {
    case MacroAssembler::Equal:
        out.print("Equal");
        return;
    case MacroAssembler::NotEqual:
        out.print("NotEqual");
        return;
    case MacroAssembler::Above:
        out.print("Above");
        return;
    case MacroAssembler::AboveOrEqual:
        out.print("AboveOrEqual");
        return;
    case MacroAssembler::Below:
        out.print("Below");
        return;
    case MacroAssembler::BelowOrEqual:
        out.print("BelowOrEqual");
        return;
    case MacroAssembler::GreaterThan:
        out.print("GreaterThan");
        return;
    case MacroAssembler::GreaterThanOrEqual:
        out.print("GreaterThanOrEqual");
        return;
    case MacroAssembler::LessThan:
        out.print("LessThan");
        return;
    case MacroAssembler::LessThanOrEqual:
        out.print("LessThanOrEqual");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, MacroAssembler::ResultCondition cond)
{
    switch (cond) {
    case MacroAssembler::Carry:
        out.print("Carry");
        return;
    case MacroAssembler::Overflow:
        out.print("Overflow");
        return;
    case MacroAssembler::Signed:
        out.print("Signed");
        return;
    case MacroAssembler::PositiveOrZero:
        out.print("PositiveOrZero");
        return;
    case MacroAssembler::Zero:
        out.print("Zero");
        return;
    case MacroAssembler::NonZero:
        out.print("NonZero");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, MacroAssembler::DoubleCondition cond)
{
    switch (cond) {
    case MacroAssembler::DoubleEqualAndOrdered:
        out.print("DoubleEqualAndOrdered");
        return;
    case MacroAssembler::DoubleNotEqualAndOrdered:
        out.print("DoubleNotEqualAndOrdered");
        return;
    case MacroAssembler::DoubleGreaterThanAndOrdered:
        out.print("DoubleGreaterThanAndOrdered");
        return;
    case MacroAssembler::DoubleGreaterThanOrEqualAndOrdered:
        out.print("DoubleGreaterThanOrEqualAndOrdered");
        return;
    case MacroAssembler::DoubleLessThanAndOrdered:
        out.print("DoubleLessThanAndOrdered");
        return;
    case MacroAssembler::DoubleLessThanOrEqualAndOrdered:
        out.print("DoubleLessThanOrEqualAndOrdered");
        return;
    case MacroAssembler::DoubleEqualOrUnordered:
        out.print("DoubleEqualOrUnordered");
        return;
    case MacroAssembler::DoubleNotEqualOrUnordered:
        out.print("DoubleNotEqualOrUnordered");
        return;
    case MacroAssembler::DoubleGreaterThanOrUnordered:
        out.print("DoubleGreaterThanOrUnordered");
        return;
    case MacroAssembler::DoubleGreaterThanOrEqualOrUnordered:
        out.print("DoubleGreaterThanOrEqualOrUnordered");
        return;
    case MacroAssembler::DoubleLessThanOrUnordered:
        out.print("DoubleLessThanOrUnordered");
        return;
    case MacroAssembler::DoubleLessThanOrEqualOrUnordered:
        out.print("DoubleLessThanOrEqualOrUnordered");
        return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(ASSEMBLER)

