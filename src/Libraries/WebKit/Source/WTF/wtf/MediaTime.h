/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

#include <wtf/FastMalloc.h>
#include <wtf/JSONValues.h>
#include <wtf/Seconds.h>
#include <wtf/text/WTFString.h>

#include <cmath>
#include <limits>
#include <math.h>
#include <stdint.h>

namespace WTF {

class PrintStream;

class WTF_EXPORT_PRIVATE MediaTime final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    enum {
        Valid = 1 << 0,
        HasBeenRounded = 1 << 1,
        PositiveInfinite = 1 << 2,
        NegativeInfinite = 1 << 3,
        Indefinite = 1 << 4,
        DoubleValue = 1 << 5,
    };

    constexpr MediaTime();
    constexpr MediaTime(int64_t value, uint32_t scale, uint8_t flags = Valid);
    MediaTime(const MediaTime&) = default;

    static MediaTime createWithFloat(float floatTime);
    static MediaTime createWithFloat(float floatTime, uint32_t timeScale);
    static MediaTime createWithDouble(double doubleTime);
    static MediaTime createWithDouble(double doubleTime, uint32_t timeScale);
    static MediaTime createWithSeconds(Seconds seconds) { return createWithDouble(seconds.value()); }

    float toFloat() const;
    double toDouble() const;
    int64_t toMicroseconds() const;

    MediaTime& operator=(const MediaTime&) = default;
    MediaTime& operator+=(const MediaTime& rhs) { return *this = *this + rhs; }
    MediaTime& operator-=(const MediaTime& rhs) { return *this = *this - rhs; }
    MediaTime operator+(const MediaTime& rhs) const;
    MediaTime operator-(const MediaTime& rhs) const;
    MediaTime operator-() const;
    MediaTime operator*(int32_t) const;
    bool operator<(const MediaTime& rhs) const { return compare(rhs) == LessThan; }
    bool operator>(const MediaTime& rhs) const { return compare(rhs) == GreaterThan; }
    bool operator==(const MediaTime& rhs) const { return compare(rhs) == EqualTo; }
    bool operator>=(const MediaTime& rhs) const { return compare(rhs) >= EqualTo; }
    bool operator<=(const MediaTime& rhs) const { return compare(rhs) <= EqualTo; }
    bool operator!() const;
    explicit operator bool() const;

    typedef enum {
        LessThan = -1,
        EqualTo = 0,
        GreaterThan = 1,
    } ComparisonFlags;

    ComparisonFlags compare(const MediaTime& rhs) const;
    bool isBetween(const MediaTime&, const MediaTime&) const;

    bool isValid() const { return m_timeFlags & Valid; }
    bool isInvalid() const { return !isValid(); }
    bool hasBeenRounded() const { return m_timeFlags & HasBeenRounded; }
    bool isPositiveInfinite() const { return m_timeFlags & PositiveInfinite; }
    bool isNegativeInfinite() const { return m_timeFlags & NegativeInfinite; }
    bool isIndefinite() const { return m_timeFlags & Indefinite; }
    bool isFinite() const { return !isInvalid() && !isIndefinite() && !isPositiveInfinite() && !isNegativeInfinite(); }
    bool hasDoubleValue() const { return m_timeFlags & DoubleValue; }
    uint8_t timeFlags() const { return m_timeFlags; }

    static const MediaTime& zeroTime();
    static const MediaTime& invalidTime();
    static const MediaTime& positiveInfiniteTime();
    static const MediaTime& negativeInfiniteTime();
    static const MediaTime& indefiniteTime();

    const int64_t& timeValue() const { return m_timeValue; }
    const uint32_t& timeScale() const { return m_timeScale; }

    void dump(PrintStream& out) const;
    String toString() const;
    String toJSONString() const;
    Ref<JSON::Object> toJSONObject() const;

    // Make the following casts errors:
    operator double() const = delete;
    MediaTime(double) = delete;
    operator int() const = delete;
    MediaTime(int) = delete;

    friend WTF_EXPORT_PRIVATE MediaTime abs(const MediaTime& rhs);

    static const uint32_t DefaultTimeScale = 10000000;
    static const uint32_t MaximumTimeScale;

    enum class RoundingFlags {
        HalfAwayFromZero = 0,
        TowardZero,
        AwayFromZero,
        TowardPositiveInfinity,
        TowardNegativeInfinity,
    };

    MediaTime toTimeScale(uint32_t, RoundingFlags = RoundingFlags::HalfAwayFromZero) const;
    MediaTime isolatedCopy() const;

private:
    void setTimeScale(uint32_t, RoundingFlags = RoundingFlags::HalfAwayFromZero);

    union {
        int64_t m_timeValue;
        double m_timeValueAsDouble;
    };
    uint32_t m_timeScale;
    uint8_t m_timeFlags;
};

constexpr MediaTime::MediaTime()
    : m_timeValue(0)
    , m_timeScale(DefaultTimeScale)
    , m_timeFlags(Valid)
{
}

constexpr MediaTime::MediaTime(int64_t value, uint32_t scale, uint8_t flags)
    : m_timeValue(value)
    , m_timeScale(scale)
    , m_timeFlags(flags)
{
    if (scale || !(flags & Valid))
        return;

    if (value < 0) {
        // Negative infinite time
        m_timeValue = -1;
        m_timeScale = 1;
        m_timeFlags = NegativeInfinite | Valid;
    } else {
        // Positive infinite time
        m_timeValue = 0;
        m_timeScale = 1;
        m_timeFlags = PositiveInfinite | Valid;
    }
}

inline MediaTime operator*(int32_t lhs, const MediaTime& rhs) { return rhs.operator*(lhs); }

WTF_EXPORT_PRIVATE extern MediaTime abs(const MediaTime& rhs);

struct WTF_EXPORT_PRIVATE MediaTimeRange {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    String toJSONString() const;

    const MediaTime start;
    const MediaTime end;
};

template<typename> struct LogArgument;

template<> struct LogArgument<MediaTime> {
    static String toString(const MediaTime& time) { return time.toJSONString(); }
};
template<> struct LogArgument<MediaTimeRange> {
    static String toString(const MediaTimeRange& range) { return range.toJSONString(); }
};

WTF_EXPORT_PRIVATE TextStream& operator<<(TextStream&, const MediaTime&);

}

using WTF::MediaTime;
using WTF::MediaTimeRange;
using WTF::abs;
