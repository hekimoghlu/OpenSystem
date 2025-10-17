/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
#include "B3ConstFloatValue.h"

#if ENABLE(B3_JIT)

#include "B3ConstDoubleValue.h"
#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

ConstFloatValue::~ConstFloatValue() = default;

Value* ConstFloatValue::negConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), -m_value);
}

Value* ConstFloatValue::addConstant(Procedure& proc, int32_t other) const
{
    return proc.add<ConstFloatValue>(origin(), m_value + static_cast<float>(other));
}

Value* ConstFloatValue::addConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    return proc.add<ConstFloatValue>(origin(), m_value + other->asFloat());
}

Value* ConstFloatValue::subConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    return proc.add<ConstFloatValue>(origin(), m_value - other->asFloat());
}

Value* ConstFloatValue::mulConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    return proc.add<ConstFloatValue>(origin(), m_value * other->asFloat());
}

Value* ConstFloatValue::bitAndConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    float result = std::bit_cast<float>(std::bit_cast<uint32_t>(m_value) & std::bit_cast<uint32_t>(other->asFloat()));
    return proc.add<ConstFloatValue>(origin(), result);
}

Value* ConstFloatValue::bitOrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    float result = std::bit_cast<float>(std::bit_cast<uint32_t>(m_value) | std::bit_cast<uint32_t>(other->asFloat()));
    return proc.add<ConstFloatValue>(origin(), result);
}

Value* ConstFloatValue::bitXorConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    float result = std::bit_cast<float>(std::bit_cast<uint32_t>(m_value) ^ std::bit_cast<uint32_t>(other->asFloat()));
    return proc.add<ConstFloatValue>(origin(), result);
}

Value* ConstFloatValue::bitwiseCastConstant(Procedure& proc) const
{
    return proc.add<Const32Value>(origin(), std::bit_cast<int32_t>(m_value));
}

Value* ConstFloatValue::floatToDoubleConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), static_cast<double>(m_value));
}

Value* ConstFloatValue::absConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), static_cast<float>(std::abs(m_value)));
}

Value* ConstFloatValue::ceilConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), ceilf(m_value));
}

Value* ConstFloatValue::floorConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), floorf(m_value));
}

Value* ConstFloatValue::sqrtConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), static_cast<float>(sqrt(m_value)));
}

Value* ConstFloatValue::divConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    return proc.add<ConstFloatValue>(origin(), m_value / other->asFloat());
}

Value* ConstFloatValue::fMinConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    return proc.add<ConstFloatValue>(origin(), Math::fMin(m_value, other->asFloat()));
}

Value* ConstFloatValue::fMaxConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasFloat())
        return nullptr;
    return proc.add<ConstFloatValue>(origin(), Math::fMax(m_value, other->asFloat()));
}

TriState ConstFloatValue::equalConstant(const Value* other) const
{
    if (!other->hasFloat())
        return TriState::Indeterminate;
    return triState(m_value == other->asFloat());
}

TriState ConstFloatValue::notEqualConstant(const Value* other) const
{
    if (!other->hasFloat())
        return TriState::Indeterminate;
    return triState(m_value != other->asFloat());
}

TriState ConstFloatValue::lessThanConstant(const Value* other) const
{
    if (!other->hasFloat())
        return TriState::Indeterminate;
    return triState(m_value < other->asFloat());
}

TriState ConstFloatValue::greaterThanConstant(const Value* other) const
{
    if (!other->hasFloat())
        return TriState::Indeterminate;
    return triState(m_value > other->asFloat());
}

TriState ConstFloatValue::lessEqualConstant(const Value* other) const
{
    if (!other->hasFloat())
        return TriState::Indeterminate;
    return triState(m_value <= other->asFloat());
}

TriState ConstFloatValue::greaterEqualConstant(const Value* other) const
{
    if (!other->hasFloat())
        return TriState::Indeterminate;
    return triState(m_value >= other->asFloat());
}

TriState ConstFloatValue::equalOrUnorderedConstant(const Value* other) const
{
    if (std::isnan(m_value))
        return TriState::True;

    if (!other->hasFloat())
        return TriState::Indeterminate;
    float otherValue = other->asFloat();
    return triState(std::isunordered(m_value, otherValue) || m_value == otherValue);
}

void ConstFloatValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    out.print(comma);
    out.printf("%le(%u)", m_value, std::bit_cast<uint32_t>(m_value));
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
