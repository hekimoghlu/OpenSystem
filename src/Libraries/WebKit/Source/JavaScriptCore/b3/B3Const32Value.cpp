/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#include "B3Const32Value.h"

#if ENABLE(B3_JIT)

#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

Const32Value::~Const32Value() = default;

Value* Const32Value::negConstant(Procedure& proc) const
{
    return proc.add<Const32Value>(origin(), -m_value);
}

Value* Const32Value::addConstant(Procedure& proc, int32_t other) const
{
    return proc.add<Const32Value>(origin(), m_value + other);
}

Value* Const32Value::addConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), m_value + other->asInt32());
}

Value* Const32Value::subConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), m_value - other->asInt32());
}

Value* Const32Value::mulConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), m_value * other->asInt32());
}

Value* Const32Value::checkAddConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    CheckedInt32 result = CheckedInt32(m_value) + CheckedInt32(other->asInt32());
    if (result.hasOverflowed())
        return nullptr;
    return proc.add<Const32Value>(origin(), result);
}

Value* Const32Value::checkSubConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    CheckedInt32 result = CheckedInt32(m_value) - CheckedInt32(other->asInt32());
    if (result.hasOverflowed())
        return nullptr;
    return proc.add<Const32Value>(origin(), result);
}

Value* Const32Value::checkMulConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    CheckedInt32 result = CheckedInt32(m_value) * CheckedInt32(other->asInt32());
    if (result.hasOverflowed())
        return nullptr;
    return proc.add<Const32Value>(origin(), result);
}

Value* Const32Value::checkNegConstant(Procedure& proc) const
{
    if (m_value == std::numeric_limits<int32_t>::min())
        return nullptr;
    return negConstant(proc);
}

Value* Const32Value::divConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), chillDiv(m_value, other->asInt32()));
}

Value* Const32Value::uDivConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), chillUDiv(m_value, other->asInt32()));
}

Value* Const32Value::modConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), chillMod(m_value, other->asInt32()));
}

Value* Const32Value::uModConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), chillUMod(m_value, other->asInt32()));
}

Value* Const32Value::bitAndConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), m_value & other->asInt32());
}

Value* Const32Value::bitOrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), m_value | other->asInt32());
}

Value* Const32Value::bitXorConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), m_value ^ other->asInt32());
}

Value* Const32Value::shlConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), m_value << (other->asInt32() & 31));
}

Value* Const32Value::sShrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), m_value >> (other->asInt32() & 31));
}

Value* Const32Value::zShrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), static_cast<int32_t>(static_cast<uint32_t>(m_value) >> (other->asInt32() & 31)));
}

Value* Const32Value::rotRConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), rotateRight(m_value, other->asInt32()));
}

Value* Const32Value::rotLConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const32Value>(origin(), rotateLeft(m_value, other->asInt32()));
}

Value* Const32Value::bitwiseCastConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), std::bit_cast<float>(m_value));
}

Value* Const32Value::iToDConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), static_cast<double>(m_value));
}

Value* Const32Value::iToFConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), static_cast<float>(m_value));
}

TriState Const32Value::equalConstant(const Value* other) const
{
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(m_value == other->asInt32());
}

TriState Const32Value::notEqualConstant(const Value* other) const
{
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(m_value != other->asInt32());
}

TriState Const32Value::lessThanConstant(const Value* other) const
{
    // INT32_MAX < x is always false.
    if (static_cast<int32_t>(m_value) == std::numeric_limits<int32_t>::max())
        return TriState::False;
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(m_value < other->asInt32());
}

TriState Const32Value::greaterThanConstant(const Value* other) const
{
    // INT32_MIN > x is always false.
    if (static_cast<int32_t>(m_value) == std::numeric_limits<int32_t>::min())
        return TriState::False;
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(m_value > other->asInt32());
}

TriState Const32Value::lessEqualConstant(const Value* other) const
{
    // INT32_MIN <= x is always true.
    if (static_cast<int32_t>(m_value) == std::numeric_limits<int32_t>::min())
        return TriState::True;
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(m_value <= other->asInt32());
}

TriState Const32Value::greaterEqualConstant(const Value* other) const
{
    // INT32_MAX >= x is always true.
    if (static_cast<int32_t>(m_value) == std::numeric_limits<int32_t>::max())
        return TriState::True;
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(m_value >= other->asInt32());
}

TriState Const32Value::aboveConstant(const Value* other) const
{
    // UINT32_MIN(0) > x is always false.
    if (static_cast<uint32_t>(m_value) == std::numeric_limits<uint32_t>::min())
        return TriState::False;
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(static_cast<uint32_t>(m_value) > static_cast<uint32_t>(other->asInt32()));
}

TriState Const32Value::belowConstant(const Value* other) const
{
    // UINT32_MAX < x is always false.
    if (static_cast<uint32_t>(m_value) == std::numeric_limits<uint32_t>::max())
        return TriState::False;
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(static_cast<uint32_t>(m_value) < static_cast<uint32_t>(other->asInt32()));
}

TriState Const32Value::aboveEqualConstant(const Value* other) const
{
    // UINT32_MAX >= x is always true.
    if (static_cast<uint32_t>(m_value) == std::numeric_limits<uint32_t>::max())
        return TriState::True;
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(static_cast<uint32_t>(m_value) >= static_cast<uint32_t>(other->asInt32()));
}

TriState Const32Value::belowEqualConstant(const Value* other) const
{
    // UINT32_MIN(0) <= x is always true.
    if (static_cast<uint32_t>(m_value) == std::numeric_limits<uint32_t>::min())
        return TriState::True;
    if (!other->hasInt32())
        return TriState::Indeterminate;
    return triState(static_cast<uint32_t>(m_value) <= static_cast<uint32_t>(other->asInt32()));
}

void Const32Value::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    out.print(comma, m_value);
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
