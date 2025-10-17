/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
#include <wtf/TimeWithDynamicClockType.h>

#include <cmath>
#include <wtf/Condition.h>
#include <wtf/PrintStream.h>
#include <wtf/Lock.h>

namespace WTF {

TimeWithDynamicClockType TimeWithDynamicClockType::now(ClockType type)
{
    switch (type) {
    case ClockType::Wall:
        return WallTime::now();
    case ClockType::Monotonic:
        return MonotonicTime::now();
    case ClockType::Approximate:
        return ApproximateTime::now();
    case ClockType::Continuous:
        return ContinuousTime::now();
    case ClockType::ContinuousApproximate:
        return ContinuousApproximateTime::now();
    }
    RELEASE_ASSERT_NOT_REACHED();
    return TimeWithDynamicClockType();
}

TimeWithDynamicClockType TimeWithDynamicClockType::nowWithSameClock() const
{
    return now(clockType());
}

WallTime TimeWithDynamicClockType::wallTime() const
{
    RELEASE_ASSERT(m_type == ClockType::Wall);
    return WallTime::fromRawSeconds(m_value);
}

MonotonicTime TimeWithDynamicClockType::monotonicTime() const
{
    RELEASE_ASSERT(m_type == ClockType::Monotonic);
    return MonotonicTime::fromRawSeconds(m_value);
}

ApproximateTime TimeWithDynamicClockType::approximateTime() const
{
    RELEASE_ASSERT(m_type == ClockType::Approximate);
    return ApproximateTime::fromRawSeconds(m_value);
}

ContinuousTime TimeWithDynamicClockType::continuousTime() const
{
    RELEASE_ASSERT(m_type == ClockType::Continuous);
    return ContinuousTime::fromRawSeconds(m_value);
}

ContinuousApproximateTime TimeWithDynamicClockType::continuousApproximateTime() const
{
    RELEASE_ASSERT(m_type == ClockType::ContinuousApproximate);
    return ContinuousApproximateTime::fromRawSeconds(m_value);
}

WallTime TimeWithDynamicClockType::approximateWallTime() const
{
    switch (m_type) {
    case ClockType::Wall:
        return wallTime();
    case ClockType::Monotonic:
        return monotonicTime().approximateWallTime();
    case ClockType::Approximate:
        return approximateTime().approximateWallTime();
    case ClockType::Continuous:
        return continuousTime().approximateWallTime();
    case ClockType::ContinuousApproximate:
        return ContinuousApproximateTime().approximateWallTime();
    }
    RELEASE_ASSERT_NOT_REACHED();
    return WallTime();
}

MonotonicTime TimeWithDynamicClockType::approximateMonotonicTime() const
{
    switch (m_type) {
    case ClockType::Wall:
        return wallTime().approximateMonotonicTime();
    case ClockType::Monotonic:
        return monotonicTime();
    case ClockType::Approximate:
        return approximateTime().approximateMonotonicTime();
    case ClockType::Continuous:
        return continuousTime().approximateMonotonicTime();
    case ClockType::ContinuousApproximate:
        return ContinuousApproximateTime().approximateMonotonicTime();
    }
    RELEASE_ASSERT_NOT_REACHED();
    return MonotonicTime();
}

Seconds TimeWithDynamicClockType::operator-(const TimeWithDynamicClockType& other) const
{
    RELEASE_ASSERT(m_type == other.m_type);
    return Seconds(m_value - other.m_value);
}

bool TimeWithDynamicClockType::operator<(const TimeWithDynamicClockType& other) const
{
    RELEASE_ASSERT(m_type == other.m_type);
    return m_value < other.m_value;
}

bool TimeWithDynamicClockType::operator>(const TimeWithDynamicClockType& other) const
{
    RELEASE_ASSERT(m_type == other.m_type);
    return m_value > other.m_value;
}

bool TimeWithDynamicClockType::operator<=(const TimeWithDynamicClockType& other) const
{
    RELEASE_ASSERT(m_type == other.m_type);
    return m_value <= other.m_value;
}

bool TimeWithDynamicClockType::operator>=(const TimeWithDynamicClockType& other) const
{
    RELEASE_ASSERT(m_type == other.m_type);
    return m_value >= other.m_value;
}

void TimeWithDynamicClockType::dump(PrintStream& out) const
{
    out.print(m_type, "(", m_value, " sec)");
}

void sleep(const TimeWithDynamicClockType& time)
{
    Lock fakeLock;
    Condition fakeCondition;
    Locker fakeLocker { fakeLock };
    fakeCondition.waitUntil(fakeLock, time);
}

bool hasElapsed(const TimeWithDynamicClockType& time)
{
    // Avoid doing now().
    if (!(time > time.withSameClockAndRawSeconds(0)))
        return true;
    if (time.secondsSinceEpoch().isInfinity())
        return false;
    
    return time <= time.nowWithSameClock();
}

} // namespace WTF


