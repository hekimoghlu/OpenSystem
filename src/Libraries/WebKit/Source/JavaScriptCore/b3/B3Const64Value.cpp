/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
#include "B3Const64Value.h"

#if ENABLE(B3_JIT)

#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

Const64Value::~Const64Value() = default;

Value* Const64Value::negConstant(Procedure& proc) const
{
    return proc.add<Const64Value>(origin(), -m_value);
}

Value* Const64Value::addConstant(Procedure& proc, int32_t other) const
{
    return proc.add<Const64Value>(origin(), m_value + static_cast<int64_t>(other));
}

Value* Const64Value::addConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), m_value + other->asInt64());
}

Value* Const64Value::subConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), m_value - other->asInt64());
}

Value* Const64Value::mulConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), m_value * other->asInt64());
}

Value* Const64Value::checkAddConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    CheckedInt64 result = CheckedInt64(m_value) + CheckedInt64(other->asInt64());
    if (result.hasOverflowed())
        return nullptr;
    return proc.add<Const64Value>(origin(), result);
}

Value* Const64Value::checkSubConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    CheckedInt64 result = CheckedInt64(m_value) - CheckedInt64(other->asInt64());
    if (result.hasOverflowed())
        return nullptr;
    return proc.add<Const64Value>(origin(), result);
}

Value* Const64Value::checkMulConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    CheckedInt64 result = CheckedInt64(m_value) * CheckedInt64(other->asInt64());
    if (result.hasOverflowed())
        return nullptr;
    return proc.add<Const64Value>(origin(), result);
}

Value* Const64Value::checkNegConstant(Procedure& proc) const
{
    if (m_value == std::numeric_limits<int64_t>::min())
        return nullptr;
    return negConstant(proc);
}

Value* Const64Value::divConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), chillDiv(m_value, other->asInt64()));
}

Value* Const64Value::uDivConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), chillUDiv(m_value, other->asInt64()));
}

Value* Const64Value::modConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), chillMod(m_value, other->asInt64()));
}

Value* Const64Value::uModConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), chillUMod(m_value, other->asInt64()));
}

Value* Const64Value::bitAndConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), m_value & other->asInt64());
}

Value* Const64Value::bitOrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), m_value | other->asInt64());
}

Value* Const64Value::bitXorConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt64())
        return nullptr;
    return proc.add<Const64Value>(origin(), m_value ^ other->asInt64());
}

Value* Const64Value::shlConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const64Value>(origin(), m_value << (other->asInt32() & 63));
}

Value* Const64Value::sShrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const64Value>(origin(), m_value >> (other->asInt32() & 63));
}

Value* Const64Value::zShrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const64Value>(origin(), static_cast<int64_t>(static_cast<uint64_t>(m_value) >> (other->asInt32() & 63)));
}

Value* Const64Value::rotRConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const64Value>(origin(), rotateRight(m_value, other->asInt32()));
}

Value* Const64Value::rotLConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasInt32())
        return nullptr;
    return proc.add<Const64Value>(origin(), rotateLeft(m_value, other->asInt32()));
}

Value* Const64Value::bitwiseCastConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), std::bit_cast<double>(m_value));
}

Value* Const64Value::iToDConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), static_cast<double>(m_value));
}

Value* Const64Value::iToFConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), static_cast<float>(m_value));
}

TriState Const64Value::equalConstant(const Value* other) const
{
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(m_value == other->asInt64());
}

TriState Const64Value::notEqualConstant(const Value* other) const
{
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(m_value != other->asInt64());
}

TriState Const64Value::lessThanConstant(const Value* other) const
{
    // INT64_MAX < x is always false.
    if (static_cast<int64_t>(m_value) == std::numeric_limits<int64_t>::max())
        return TriState::False;
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(m_value < other->asInt64());
}

TriState Const64Value::greaterThanConstant(const Value* other) const
{
    // INT64_MIN > x is always false.
    if (static_cast<int64_t>(m_value) == std::numeric_limits<int64_t>::min())
        return TriState::False;
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(m_value > other->asInt64());
}

TriState Const64Value::lessEqualConstant(const Value* other) const
{
    // INT64_MIN <= x is always true.
    if (static_cast<int64_t>(m_value) == std::numeric_limits<int64_t>::min())
        return TriState::True;
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(m_value <= other->asInt64());
}

TriState Const64Value::greaterEqualConstant(const Value* other) const
{
    // INT64_MAX >= x is always true.
    if (static_cast<int64_t>(m_value) == std::numeric_limits<int64_t>::max())
        return TriState::True;
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(m_value >= other->asInt64());
}

TriState Const64Value::aboveConstant(const Value* other) const
{
    // UINT64_MIN > x is always false.
    if (static_cast<uint64_t>(m_value) == std::numeric_limits<uint64_t>::min())
        return TriState::False;
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(static_cast<uint64_t>(m_value) > static_cast<uint64_t>(other->asInt64()));
}

TriState Const64Value::belowConstant(const Value* other) const
{
    // UINT64_MAX < x is always false.
    if (static_cast<uint64_t>(m_value) == std::numeric_limits<uint64_t>::max())
        return TriState::False;
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(static_cast<uint64_t>(m_value) < static_cast<uint64_t>(other->asInt64()));
}

TriState Const64Value::aboveEqualConstant(const Value* other) const
{
    // UINT64_MAX >= x is always true.
    if (static_cast<uint64_t>(m_value) == std::numeric_limits<uint64_t>::max())
        return TriState::True;
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(static_cast<uint64_t>(m_value) >= static_cast<uint64_t>(other->asInt64()));
}

TriState Const64Value::belowEqualConstant(const Value* other) const
{
    // UINT64_MIN <= x is always true.
    if (static_cast<uint64_t>(m_value) == std::numeric_limits<uint64_t>::min())
        return TriState::True;
    if (!other->hasInt64())
        return TriState::Indeterminate;
    return triState(static_cast<uint64_t>(m_value) <= static_cast<uint64_t>(other->asInt64()));
}

void Const64Value::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    out.print(comma, m_value);
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
