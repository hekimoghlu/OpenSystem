/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include "AirArg.h"

#if ENABLE(B3_JIT)

#include "AirSpecial.h"
#include "AirStackSlot.h"
#include "B3Value.h"
#include "GPRInfo.h"

#if !ASSERT_ENABLED
IGNORE_RETURN_TYPE_WARNINGS_BEGIN
#endif

namespace JSC { namespace B3 { namespace Air {

bool Arg::isStackMemory() const
{
    switch (kind()) {
    case Addr:
        return base() == Air::Tmp(GPRInfo::callFrameRegister)
            || base() == Air::Tmp(MacroAssembler::stackPointerRegister);
    case ExtendedOffsetAddr:
    case Stack:
    case CallArg:
        return true;
    default:
        return false;
    }
}

bool Arg::isRepresentableAs(Width width, Signedness signedness) const
{
    return isRepresentableAs(width, signedness, value());
}

bool Arg::usesTmp(Air::Tmp tmp) const
{
    bool uses = false;
    const_cast<Arg*>(this)->forEachTmpFast(
        [&] (Air::Tmp otherTmp) {
            if (otherTmp == tmp)
                uses = true;
        });
    return uses;
}

bool Arg::canRepresent(Type type) const
{
    return isBank(bankForType(type));
}

bool Arg::canRepresent(Value* value) const
{
    return canRepresent(value->type());
}

bool Arg::isCompatibleBank(const Arg& other) const
{
    if (hasBank())
        return other.isBank(bank());
    if (other.hasBank())
        return isBank(other.bank());
    return true;
}

unsigned Arg::jsHash() const
{
    unsigned result = static_cast<unsigned>(m_kind);
    
    switch (m_kind) {
    case Invalid:
    case Special:
    case SIMDInfo:
        break;
    case Tmp:
        result += m_base.internalValue();
        break;
    case Imm:
    case BitImm:
    case FPImm32:
    case ZeroReg:
    case CallArg:
    case RelCond:
    case ResCond:
    case DoubleCond:
    case StatusCond:
    case WidthArg:
        result += static_cast<unsigned>(m_offset);
        break;
    case BigImm:
    case BitImm64:
    case FPImm64:
        result += static_cast<unsigned>(m_offset);
        result += static_cast<unsigned>(m_offset >> 32);
        break;
    case SimpleAddr:
        result += m_base.internalValue();
        break;
    case Addr:
    case ExtendedOffsetAddr:
        result += m_offset;
        result += m_base.internalValue();
        break;
    case Index:
        result += static_cast<unsigned>(m_offset);
        result += m_scale;
        result += m_base.internalValue();
        result += m_index.internalValue();
        break;
    case PreIndex:
    case PostIndex:
        result += m_offset;
        result += m_base.internalValue();
        break;
    case Stack:
        result += static_cast<unsigned>(m_scale);
        result += stackSlot()->index();
        break;
    }
    
    return result;
}

void Arg::dump(PrintStream& out) const
{
    switch (m_kind) {
    case Invalid:
        out.print("<invalid>");
        return;
    case Tmp:
        out.print(tmp());
        return;
    case Imm:
        out.print("$", m_offset);
        return;
    case BigImm:
        out.printf("$0x%llx", static_cast<long long unsigned>(m_offset));
        return;
    case BitImm:
        out.print("$", m_offset);
        return;
    case BitImm64:
        out.printf("$0x%llx", static_cast<long long unsigned>(m_offset));
        return;
    case FPImm32:
        out.print("$", m_offset);
        return;
    case FPImm64:
        out.printf("$0x%llx", static_cast<long long unsigned>(m_offset));
        return;
    case ZeroReg:
        out.print("%xzr");
        return;
    case SimpleAddr:
        out.print("(", base(), ")");
        return;
    case Addr:
    case ExtendedOffsetAddr:
        if (offset())
            out.print(offset());
        out.print("(", base(), ")");
        return;
    case Index:
        if (offset())
            out.print(offset());
        out.print("(", base(), ",", index());
        if (scale() != 1)
            out.print(",", scale());
        out.print(")");
        return;
    case PreIndex:
        out.print("(", base(), ",Pre($", offset(), "))");
        return;
    case PostIndex:
        out.print("(", base(), ",Post($", offset(), "))");
        return;
    case Stack:
        if (offset())
            out.print(offset());
        out.print("(", pointerDump(stackSlot()), ")");
        return;
    case CallArg:
        if (offset())
            out.print(offset());
        out.print("(callArg)");
        return;
    case RelCond:
        out.print(asRelationalCondition());
        return;
    case ResCond:
        out.print(asResultCondition());
        return;
    case DoubleCond:
        out.print(asDoubleCondition());
        return;
    case StatusCond:
        out.print(asStatusCondition());
        return;
    case Special:
        out.print(pointerDump(special()));
        return;
    case WidthArg:
        out.print(width());
        return;
    case SIMDInfo:
        out.print("{ ", simdInfo().lane, ", ", simdInfo().signMode, " }");
        return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

} } } // namespace JSC::B3::Air

namespace WTF {

using namespace JSC::B3::Air;

void printInternal(PrintStream& out, Arg::Kind kind)
{
    switch (kind) {
    case Arg::Invalid:
        out.print("Invalid");
        return;
    case Arg::Tmp:
        out.print("Tmp");
        return;
    case Arg::Imm:
        out.print("Imm");
        return;
    case Arg::BigImm:
        out.print("BigImm");
        return;
    case Arg::BitImm:
        out.print("BitImm");
        return;
    case Arg::BitImm64:
        out.print("BitImm64");
        return;
    case Arg::FPImm32:
        out.print("FPImm32");
        return;
    case Arg::FPImm64:
        out.print("FPImm64");
        return;
    case Arg::ZeroReg:
        out.print("ZeroReg");
        return;
    case Arg::SimpleAddr:
        out.print("SimpleAddr");
        return;
    case Arg::Addr:
        out.print("Addr");
        return;
    case Arg::ExtendedOffsetAddr:
        out.print("ExtendedOffsetAddr");
        return;
    case Arg::Stack:
        out.print("Stack");
        return;
    case Arg::CallArg:
        out.print("CallArg");
        return;
    case Arg::Index:
        out.print("Index");
        return;
    case Arg::PreIndex:
        out.print("PreIndex");
        return;
    case Arg::PostIndex:
        out.print("PostIndex");
        return;
    case Arg::RelCond:
        out.print("RelCond");
        return;
    case Arg::ResCond:
        out.print("ResCond");
        return;
    case Arg::DoubleCond:
        out.print("DoubleCond");
        return;
    case Arg::StatusCond:
        out.print("StatusCond");
        return;
    case Arg::Special:
        out.print("Special");
        return;
    case Arg::WidthArg:
        out.print("WidthArg");
        return;
    case Arg::SIMDInfo:
        out.print("SIMDInfo");
        return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, Arg::Temperature temperature)
{
    switch (temperature) {
    case Arg::Cold:
        out.print("Cold");
        return;
    case Arg::Warm:
        out.print("Warm");
        return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, Arg::Phase phase)
{
    switch (phase) {
    case Arg::Early:
        out.print("Early");
        return;
    case Arg::Late:
        out.print("Late");
        return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, Arg::Timing timing)
{
    switch (timing) {
    case Arg::OnlyEarly:
        out.print("OnlyEarly");
        return;
    case Arg::OnlyLate:
        out.print("OnlyLate");
        return;
    case Arg::EarlyAndLate:
        out.print("EarlyAndLate");
        return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, Arg::Role role)
{
    switch (role) {
    case Arg::Use:
        out.print("Use");
        return;
    case Arg::Def:
        out.print("Def");
        return;
    case Arg::UseDef:
        out.print("UseDef");
        return;
    case Arg::ZDef:
        out.print("ZDef");
        return;
    case Arg::UseZDef:
        out.print("UseZDef");
        return;
    case Arg::UseAddr:
        out.print("UseAddr");
        return;
    case Arg::ColdUse:
        out.print("ColdUse");
        return;
    case Arg::LateUse:
        out.print("LateUse");
        return;
    case Arg::LateColdUse:
        out.print("LateColdUse");
        return;
    case Arg::EarlyDef:
        out.print("EarlyDef");
        return;
    case Arg::EarlyZDef:
        out.print("EarlyZDef");
        return;
    case Arg::Scratch:
        out.print("Scratch");
        return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, Arg::Signedness signedness)
{
    switch (signedness) {
    case Arg::Signed:
        out.print("Signed");
        return;
    case Arg::Unsigned:
        out.print("Unsigned");
        return;
    }

    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#if !ASSERT_ENABLED
IGNORE_RETURN_TYPE_WARNINGS_END
#endif

#endif // ENABLE(B3_JIT)
