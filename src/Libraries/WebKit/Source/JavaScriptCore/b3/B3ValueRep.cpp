/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#include "B3ValueRep.h"

#if ENABLE(B3_JIT)

#include "AssemblyHelpers.h"
#include "JSCJSValueInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace B3 {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ValueRep);

void ValueRep::addUsedRegistersTo(bool isSIMDContext, RegisterSetBuilder& set) const
{
    switch (m_kind) {
    case WarmAny:
    case ColdAny:
    case LateColdAny:
    case SomeRegister:
    case SomeRegisterWithClobber:
    case SomeEarlyRegister:
    case SomeLateRegister:
    case Constant:
        return;
    case LateRegister:
    case Register:
        set.add(reg(), isSIMDContext ? conservativeWidth(reg()) : conservativeWidthWithoutVectors(reg()));
        return;
    case Stack:
    case StackArgument:
        set.add(MacroAssembler::stackPointerRegister, IgnoreVectors);
        set.add(GPRInfo::callFrameRegister, IgnoreVectors);
        return;
#if USE(JSVALUE32_64)
    case RegisterPair:
        break;
#endif
    }
    RELEASE_ASSERT_NOT_REACHED();
}

RegisterSetBuilder ValueRep::usedRegisters(bool isSIMDContext) const
{
    RegisterSetBuilder result;
    addUsedRegistersTo(isSIMDContext, result);
    return result;
}

void ValueRep::dump(PrintStream& out) const
{
    out.print(m_kind);
    switch (m_kind) {
    case WarmAny:
    case ColdAny:
    case LateColdAny:
    case SomeRegister:
    case SomeRegisterWithClobber:
    case SomeEarlyRegister:
    case SomeLateRegister:
        return;
    case LateRegister:
    case Register:
        out.print("(", reg(), ")");
        return;
#if USE(JSVALUE32_64)
    case RegisterPair:
        out.print("(", u.regPair.regLo, ", ", u.regPair.regHi, ")");
        return;
#endif
    case Stack:
        out.print("(", offsetFromFP(), ")");
        return;
    case StackArgument:
        out.print("(", offsetFromSP(), ")");
        return;
    case Constant:
        out.print("(", value(), ")");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

// We use `B3::ValueRep` for bookkeeping in the BBQ wasm backend, including on
// 32-bit platforms, but not for code generation (yet!), so we don't actually
// want to provide these symbols until they are properly supported on those
// platforms.

#if USE(JSVALUE64)

void ValueRep::emitRestore(AssemblyHelpers& jit, Reg reg) const
{
    if (reg.isGPR()) {
        switch (kind()) {
        case LateRegister:
        case Register:
            if (isGPR())
                jit.move(gpr(), reg.gpr());
            else
                jit.moveDoubleTo64(fpr(), reg.gpr());
            break;
        case Stack:
            jit.load64(AssemblyHelpers::Address(GPRInfo::callFrameRegister, offsetFromFP()), reg.gpr());
            break;
        case Constant:
            jit.move(AssemblyHelpers::TrustedImm64(value()), reg.gpr());
            break;
        default:
            RELEASE_ASSERT_NOT_REACHED();
            break;
        }
        return;
    }
    
    switch (kind()) {
    case LateRegister:
    case Register:
        if (isGPR())
            jit.move64ToDouble(gpr(), reg.fpr());
        else
            jit.moveDouble(fpr(), reg.fpr());
        break;
    case Stack:
        jit.loadDouble(AssemblyHelpers::Address(GPRInfo::callFrameRegister, offsetFromFP()), reg.fpr());
        break;
    case Constant:
        jit.move(AssemblyHelpers::TrustedImm64(value()), jit.scratchRegister());
        jit.move64ToDouble(jit.scratchRegister(), reg.fpr());
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        break;
    }
}

ValueRecovery ValueRep::recoveryForJSValue() const
{
    switch (kind()) {
    case LateRegister:
    case Register:
        return ValueRecovery::inGPR(gpr(), DataFormatJS);
    case Stack:
        RELEASE_ASSERT(!(offsetFromFP() % sizeof(EncodedJSValue)));
        return ValueRecovery::displacedInJSStack(
            VirtualRegister(offsetFromFP() / sizeof(EncodedJSValue)),
            DataFormatJS);
    case Constant:
        return ValueRecovery::constant(JSValue::decode(value()));
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return { };
    }
}

#endif // USE(JSVALUE64) [see note above]

} } // namespace JSC::B3

namespace WTF {

using namespace JSC::B3;

void printInternal(PrintStream& out, ValueRep::Kind kind)
{
    switch (kind) {
    case ValueRep::WarmAny:
        out.print("WarmAny");
        return;
    case ValueRep::ColdAny:
        out.print("ColdAny");
        return;
    case ValueRep::LateColdAny:
        out.print("LateColdAny");
        return;
    case ValueRep::SomeRegister:
        out.print("SomeRegister");
        return;
#if USE(JSVALUE32_64)
    case ValueRep::RegisterPair:
        out.print("SomeRegisterPair");
        return;
#endif
    case ValueRep::SomeRegisterWithClobber:
        out.print("SomeRegisterWithClobber");
        return;
    case ValueRep::SomeEarlyRegister:
        out.print("SomeEarlyRegister");
        return;
    case ValueRep::SomeLateRegister:
        out.print("SomeLateRegister");
        return;
    case ValueRep::Register:
        out.print("Register");
        return;
    case ValueRep::LateRegister:
        out.print("LateRegister");
        return;
    case ValueRep::Stack:
        out.print("Stack");
        return;
    case ValueRep::StackArgument:
        out.print("StackArgument");
        return;
    case ValueRep::Constant:
        out.print("Constant");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(B3_JIT)
