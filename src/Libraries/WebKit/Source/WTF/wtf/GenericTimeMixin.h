/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
#pragma once

#include <wtf/Seconds.h>

namespace WTF {

template<typename DerivedTime>
class GenericTimeMixin {
    WTF_MAKE_FAST_ALLOCATED;
public:
    // Call this if you know for sure that the double represents the time according to the
    // same time source as DerivedTime. It must be in seconds.
    static constexpr DerivedTime fromRawSeconds(double value)
    {
        return DerivedTime(value);
    }

    static constexpr DerivedTime infinity() { return fromRawSeconds(std::numeric_limits<double>::infinity()); }
    static constexpr DerivedTime nan() { return fromRawSeconds(std::numeric_limits<double>::quiet_NaN()); }

    bool isNaN() const { return std::isnan(m_value); }
    bool isInfinity() const { return std::isinf(m_value); }
    bool isFinite() const { return std::isfinite(m_value); }

    constexpr Seconds secondsSinceEpoch() const { return Seconds(m_value); }

    explicit constexpr operator bool() const { return !!m_value; }

    constexpr DerivedTime operator+(Seconds other) const
    {
        return fromRawSeconds(m_value + other.value());
    }

    constexpr DerivedTime operator-(Seconds other) const
    {
        return fromRawSeconds(m_value - other.value());
    }

    Seconds operator%(Seconds other) const
    {
        return Seconds { fmod(m_value, other.value()) };
    }

    // Time is a scalar and scalars can be negated as this could arise from algebraic
    // transformations. So, we allow it.
    constexpr DerivedTime operator-() const
    {
        return fromRawSeconds(-m_value);
    }

    DerivedTime operator+=(Seconds other)
    {
        return *static_cast<DerivedTime*>(this) = *static_cast<DerivedTime*>(this) + other;
    }

    DerivedTime operator-=(Seconds other)
    {
        return *static_cast<DerivedTime*>(this) = *static_cast<DerivedTime*>(this) - other;
    }

    constexpr Seconds operator-(DerivedTime other) const
    {
        return Seconds(m_value - other.m_value);
    }

    friend constexpr bool operator==(GenericTimeMixin, GenericTimeMixin) = default;

    constexpr bool operator<(const GenericTimeMixin& other) const
    {
        return m_value < other.m_value;
    }

    constexpr bool operator>(const GenericTimeMixin& other) const
    {
        return m_value > other.m_value;
    }

    constexpr bool operator<=(const GenericTimeMixin& other) const
    {
        return m_value <= other.m_value;
    }

    constexpr bool operator>=(const GenericTimeMixin& other) const
    {
        return m_value >= other.m_value;
    }

    DerivedTime isolatedCopy() const
    {
        return *static_cast<const DerivedTime*>(this);
    }

    static constexpr DerivedTime timePointFromNow(Seconds relativeTimeFromNow)
    {
        if (relativeTimeFromNow.isInfinity())
            return DerivedTime::fromRawSeconds(relativeTimeFromNow.value());
        return DerivedTime::now() + relativeTimeFromNow;
    }

protected:
    // This is the epoch. So, x.secondsSinceEpoch() should be the same as x - DerivedTime().
    constexpr GenericTimeMixin() = default;

    constexpr GenericTimeMixin(double rawValue)
        : m_value(rawValue)
    {
    }

    double m_value { 0 };
};

} // namespace WTF
