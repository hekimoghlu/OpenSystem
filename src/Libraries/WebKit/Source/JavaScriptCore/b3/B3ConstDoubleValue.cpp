/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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
#include "B3ConstDoubleValue.h"

#if ENABLE(B3_JIT)

#include "B3ConstFloatValue.h"
#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

ConstDoubleValue::~ConstDoubleValue() = default;

Value* ConstDoubleValue::negConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), -m_value);
}

Value* ConstDoubleValue::addConstant(Procedure& proc, int32_t other) const
{
    return proc.add<ConstDoubleValue>(origin(), m_value + static_cast<double>(other));
}

Value* ConstDoubleValue::addConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    return proc.add<ConstDoubleValue>(origin(), m_value + other->asDouble());
}

Value* ConstDoubleValue::subConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    return proc.add<ConstDoubleValue>(origin(), m_value - other->asDouble());
}

Value* ConstDoubleValue::mulConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    return proc.add<ConstDoubleValue>(origin(), m_value * other->asDouble());
}

Value* ConstDoubleValue::bitAndConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    double result = std::bit_cast<double>(std::bit_cast<uint64_t>(m_value) & std::bit_cast<uint64_t>(other->asDouble()));
    return proc.add<ConstDoubleValue>(origin(), result);
}

Value* ConstDoubleValue::bitOrConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    double result = std::bit_cast<double>(std::bit_cast<uint64_t>(m_value) | std::bit_cast<uint64_t>(other->asDouble()));
    return proc.add<ConstDoubleValue>(origin(), result);
}

Value* ConstDoubleValue::bitXorConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    double result = std::bit_cast<double>(std::bit_cast<uint64_t>(m_value) ^ std::bit_cast<uint64_t>(other->asDouble()));
    return proc.add<ConstDoubleValue>(origin(), result);
}


Value* ConstDoubleValue::bitwiseCastConstant(Procedure& proc) const
{
    return proc.add<Const64Value>(origin(), std::bit_cast<int64_t>(m_value));
}

Value* ConstDoubleValue::doubleToFloatConstant(Procedure& proc) const
{
    return proc.add<ConstFloatValue>(origin(), static_cast<float>(m_value));
}

Value* ConstDoubleValue::absConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), std::abs(m_value));
}

Value* ConstDoubleValue::ceilConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), ceil(m_value));
}

Value* ConstDoubleValue::floorConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), floor(m_value));
}

Value* ConstDoubleValue::sqrtConstant(Procedure& proc) const
{
    return proc.add<ConstDoubleValue>(origin(), sqrt(m_value));
}

Value* ConstDoubleValue::divConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    return proc.add<ConstDoubleValue>(origin(), m_value / other->asDouble());
}

Value* ConstDoubleValue::modConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    return proc.add<ConstDoubleValue>(origin(), fmod(m_value, other->asDouble()));
}

Value* ConstDoubleValue::fMinConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    return proc.add<ConstDoubleValue>(origin(), Math::fMin(m_value, other->asDouble()));
}

Value* ConstDoubleValue::fMaxConstant(Procedure& proc, const Value* other) const
{
    if (!other->hasDouble())
        return nullptr;
    return proc.add<ConstDoubleValue>(origin(), Math::fMax(m_value, other->asDouble()));
}

TriState ConstDoubleValue::equalConstant(const Value* other) const
{
    if (!other->hasDouble())
        return TriState::Indeterminate;
    return triState(m_value == other->asDouble());
}

TriState ConstDoubleValue::notEqualConstant(const Value* other) const
{
    if (!other->hasDouble())
        return TriState::Indeterminate;
    return triState(m_value != other->asDouble());
}

TriState ConstDoubleValue::lessThanConstant(const Value* other) const
{
    if (!other->hasDouble())
        return TriState::Indeterminate;
    return triState(m_value < other->asDouble());
}

TriState ConstDoubleValue::greaterThanConstant(const Value* other) const
{
    if (!other->hasDouble())
        return TriState::Indeterminate;
    return triState(m_value > other->asDouble());
}

TriState ConstDoubleValue::lessEqualConstant(const Value* other) const
{
    if (!other->hasDouble())
        return TriState::Indeterminate;
    return triState(m_value <= other->asDouble());
}

TriState ConstDoubleValue::greaterEqualConstant(const Value* other) const
{
    if (!other->hasDouble())
        return TriState::Indeterminate;
    return triState(m_value >= other->asDouble());
}

TriState ConstDoubleValue::equalOrUnorderedConstant(const Value* other) const
{
    if (std::isnan(m_value))
        return TriState::True;

    if (!other->hasDouble())
        return TriState::Indeterminate;
    double otherValue = other->asDouble();
    return triState(std::isunordered(m_value, otherValue) || m_value == otherValue);
}

void ConstDoubleValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    out.print(comma);
    out.printf("%le(%llu)", m_value, static_cast<unsigned long long>(std::bit_cast<uint64_t>(m_value)));
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
