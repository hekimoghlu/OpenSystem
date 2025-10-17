/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "WebAnimationTime.h"

#include "CSSNumericFactory.h"
#include "CSSUnitValue.h"
#include "CSSUnits.h"
#include "WebAnimationUtilities.h"

namespace WebCore {

WebAnimationTime::WebAnimationTime(std::optional<Seconds> time, std::optional<double> percentage)
{
    ASSERT(time || percentage);
    ASSERT(!!time != !!percentage);
    if (time) {
        m_type = Type::Time;
        m_value = time->seconds();
    } else {
        m_type = Type::Percentage;
        m_value = *percentage;
    }
}

WebAnimationTime::WebAnimationTime(const Seconds& value)
    : m_type(Type::Time)
    , m_value(value.seconds())
{
}

WebAnimationTime::WebAnimationTime(Type type, double value)
    : m_type(type)
    , m_value(value)
{
}

WebAnimationTime::WebAnimationTime(const CSSNumberish& value)
{
    if (auto* doubleValue = std::get_if<double>(&value)) {
        m_type = Type::Time;
        m_value = *doubleValue / 1000;
        return;
    }

    ASSERT(std::holds_alternative<RefPtr<CSSNumericValue>>(value));
    auto numericValue = std::get<RefPtr<CSSNumericValue>>(value);
    if (RefPtr unitValue = dynamicDowncast<CSSUnitValue>(numericValue.get())) {
        if (unitValue->unitEnum() == CSSUnitType::CSS_NUMBER) {
            m_type = Type::Time;
            m_value = unitValue->value() / 1000;
        } else if (auto milliseconds = unitValue->convertTo(CSSUnitType::CSS_MS)) {
            m_type = Type::Time;
            m_value = milliseconds->value() / 1000;
        } else if (auto seconds = unitValue->convertTo(CSSUnitType::CSS_S)) {
            m_type = Type::Time;
            m_value = seconds->value();
        } else if (auto percentage = unitValue->convertTo(CSSUnitType::CSS_PERCENTAGE)) {
            m_type = Type::Percentage;
            m_value = percentage->value();
        }
    }
}

WebAnimationTime WebAnimationTime::fromMilliseconds(double milliseconds)
{
    return { Type::Time, milliseconds / 1000 };
}

WebAnimationTime WebAnimationTime::fromPercentage(double percentage)
{
    return { Type::Percentage, percentage };
}

std::optional<Seconds> WebAnimationTime::time() const
{
    if (m_type == Type::Time)
        return Seconds { m_value };
    return std::nullopt;
}

std::optional<double> WebAnimationTime::percentage() const
{
    if (m_type == Type::Percentage)
        return m_value;
    return std::nullopt;
}

bool WebAnimationTime::isValid() const
{
    return m_type != Type::Unknown;
}

bool WebAnimationTime::isInfinity() const
{
    return std::isinf(m_value);
}

bool WebAnimationTime::isZero() const
{
    return !m_value;
}

bool WebAnimationTime::isNaN() const
{
    return std::isnan(m_value);
}

WebAnimationTime WebAnimationTime::matchingZero() const
{
    return { m_type, 0 };
}

WebAnimationTime WebAnimationTime::matchingInfinity() const
{
    return { m_type, std::numeric_limits<double>::infinity() };
}

WebAnimationTime WebAnimationTime::matchingEpsilon() const
{
    if (m_type == Type::Percentage)
        return WebAnimationTime::fromPercentage(0.000001);
    return { WebCore::timeEpsilon };
}

bool WebAnimationTime::approximatelyEqualTo(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    if (m_type == Type::Time)
        return std::abs(time()->microseconds() - other.time()->microseconds()) < timeEpsilon.microseconds();
    return m_value == other.m_value;
}

bool WebAnimationTime::approximatelyLessThan(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    if (m_type == Type::Time)
        return (*time() + timeEpsilon) < *other.time();
    return m_value < other.m_value;
}

bool WebAnimationTime::approximatelyGreaterThan(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    if (m_type == Type::Time)
        return (*time() - timeEpsilon) > *other.time();
    return m_value > other.m_value;
}

WebAnimationTime WebAnimationTime::operator+(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    return { m_type, m_value + other.m_value };
}

WebAnimationTime WebAnimationTime::operator-(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    return { m_type, m_value - other.m_value };
}

double WebAnimationTime::operator/(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    return m_value / other.m_value;
}

WebAnimationTime& WebAnimationTime::operator+=(const WebAnimationTime& other)
{
    ASSERT(m_type == other.m_type);
    m_value += other.m_value;
    return *this;
}

WebAnimationTime& WebAnimationTime::operator-=(const WebAnimationTime& other)
{
    ASSERT(m_type == other.m_type);
    m_value -= other.m_value;
    return *this;
}

bool WebAnimationTime::operator<(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    return m_value < other.m_value;
}

bool WebAnimationTime::operator<=(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    return m_value <= other.m_value;
}

bool WebAnimationTime::operator>(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    return m_value > other.m_value;
}

bool WebAnimationTime::operator>=(const WebAnimationTime& other) const
{
    ASSERT(m_type == other.m_type);
    return m_value >= other.m_value;
}

bool WebAnimationTime::operator==(const WebAnimationTime& other) const
{
    return m_type == other.m_type && m_value == other.m_value;
}

WebAnimationTime WebAnimationTime::operator+(const Seconds& other) const
{
    ASSERT(m_type == Type::Time);
    return { m_type, m_value + other.seconds() };
}

WebAnimationTime WebAnimationTime::operator-(const Seconds& other) const
{
    ASSERT(m_type == Type::Time);
    return { m_type, m_value - other.seconds() };
}

bool WebAnimationTime::operator<(const Seconds& other) const
{
    ASSERT(m_type == Type::Time);
    return m_value < other.seconds();
}

bool WebAnimationTime::operator<=(const Seconds& other) const
{
    ASSERT(m_type == Type::Time);
    return m_value <= other.seconds();
}

bool WebAnimationTime::operator>(const Seconds& other) const
{
    ASSERT(m_type == Type::Time);
    return m_value > other.seconds();
}

bool WebAnimationTime::operator>=(const Seconds& other) const
{
    ASSERT(m_type == Type::Time);
    return m_value >= other.seconds();
}

bool WebAnimationTime::operator==(const Seconds& other) const
{
    return m_type == Type::Time && m_value == other.seconds();
}

WebAnimationTime WebAnimationTime::operator*(double scalar) const
{
    return { m_type, m_value * scalar };
}

WebAnimationTime WebAnimationTime::operator/(double scalar) const
{
    return { m_type, m_value / scalar };
}

WebAnimationTime::operator Seconds() const
{
    ASSERT(m_type == Type::Time);
    return Seconds(m_value);
}

WebAnimationTime::operator CSSNumberish() const
{
    if (m_type == Type::Time)
        return secondsToWebAnimationsAPITime(*this);
    ASSERT(m_type == Type::Percentage);
    return CSSNumericFactory::percent(m_value);
}

void WebAnimationTime::dump(TextStream& ts) const
{
    if (m_type == Type::Time) {
        ts << m_value * 1000;
        return;
    }
    ASSERT(m_type == Type::Percentage);
    ts << m_value << "%";
    return;
}

TextStream& operator<<(TextStream& ts, const WebAnimationTime& value)
{
    value.dump(ts);
    return ts;
}

} // namespace WebCore
