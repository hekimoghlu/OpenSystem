/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#include <wtf/Seconds.h>

#include <wtf/ApproximateTime.h>
#include <wtf/Condition.h>
#include <wtf/ContinuousTime.h>
#include <wtf/Lock.h>
#include <wtf/MonotonicTime.h>
#include <wtf/PrintStream.h>
#include <wtf/TimeWithDynamicClockType.h>
#include <wtf/WallTime.h>
#include <wtf/text/TextStream.h>

namespace WTF {

WallTime Seconds::operator+(WallTime other) const
{
    return other + *this;
}

MonotonicTime Seconds::operator+(MonotonicTime other) const
{
    return other + *this;
}

ApproximateTime Seconds::operator+(ApproximateTime other) const
{
    return other + *this;
}

ContinuousTime Seconds::operator+(ContinuousTime other) const
{
    return other + *this;
}

ContinuousApproximateTime Seconds::operator+(ContinuousApproximateTime other) const
{
    return other + *this;
}

TimeWithDynamicClockType Seconds::operator+(const TimeWithDynamicClockType& other) const
{
    return other + *this;
}

WallTime Seconds::operator-(WallTime other) const
{
    return WallTime::fromRawSeconds(value() - other.secondsSinceEpoch().value());
}

MonotonicTime Seconds::operator-(MonotonicTime other) const
{
    return MonotonicTime::fromRawSeconds(value() - other.secondsSinceEpoch().value());
}

ApproximateTime Seconds::operator-(ApproximateTime other) const
{
    return ApproximateTime::fromRawSeconds(value() - other.secondsSinceEpoch().value());
}

ContinuousTime Seconds::operator-(ContinuousTime other) const
{
    return ContinuousTime::fromRawSeconds(value() - other.secondsSinceEpoch().value());
}

ContinuousApproximateTime Seconds::operator-(ContinuousApproximateTime other) const
{
    return ContinuousApproximateTime::fromRawSeconds(value() - other.secondsSinceEpoch().value());
}

TimeWithDynamicClockType Seconds::operator-(const TimeWithDynamicClockType& other) const
{
    return other.withSameClockAndRawSeconds(value() - other.secondsSinceEpoch().value());
}

void Seconds::dump(PrintStream& out) const
{
    out.print(m_value, " sec");
}

TextStream& operator<<(TextStream& ts, Seconds seconds)
{
    ts << seconds.value() << "s";
    return ts;
}

void sleep(Seconds value)
{
    // It's very challenging to find portable ways of sleeping for less than a second. On UNIX, you want to
    // use usleep() but it's hard to #include it in a portable way (you'd think it's in unistd.h, but then
    // you'd be wrong on some OSX SDKs). Also, usleep() won't save you on Windows. Hence, bottoming out in
    // lock code, which already solves the sleeping problem, is probably for the best.

    Lock fakeLock;
    Condition fakeCondition;
    Locker fakeLocker { fakeLock };
    fakeCondition.waitFor(fakeLock, value);
}

} // namespace WTF

