/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
#include "TimeRanges.h"

namespace WebCore {

Ref<TimeRanges> TimeRanges::create()
{
    return adoptRef(*new TimeRanges);
}

Ref<TimeRanges> TimeRanges::create(double start, double end)
{
    return adoptRef(*new TimeRanges(start, end));
}

Ref<TimeRanges> TimeRanges::create(const PlatformTimeRanges& other)
{
    return adoptRef(*new TimeRanges(other));
}

TimeRanges::TimeRanges()
{
}

TimeRanges::TimeRanges(double start, double end)
    : m_ranges(PlatformTimeRanges(MediaTime::createWithDouble(start), MediaTime::createWithDouble(end)))
{
}

TimeRanges::TimeRanges(const PlatformTimeRanges& other)
    : m_ranges(other)
{
}

ExceptionOr<double> TimeRanges::start(unsigned index) const
{
    bool valid;
    MediaTime result = m_ranges.start(index, valid);
    if (!valid)
        return Exception { ExceptionCode::IndexSizeError };
    return result.toDouble();
}

ExceptionOr<double> TimeRanges::end(unsigned index) const
{ 
    bool valid;
    MediaTime result = m_ranges.end(index, valid);
    if (!valid)
        return Exception { ExceptionCode::IndexSizeError };
    return result.toDouble();
}

void TimeRanges::invert()
{
    m_ranges.invert();
}

Ref<TimeRanges> TimeRanges::copy() const
{
    return TimeRanges::create(m_ranges);
}

void TimeRanges::intersectWith(const TimeRanges& other)
{
    m_ranges.intersectWith(other.ranges());
}

void TimeRanges::unionWith(const TimeRanges& other)
{
    m_ranges.unionWith(other.ranges());
}

unsigned TimeRanges::length() const
{
    return m_ranges.length();
}

void TimeRanges::add(double start, double end, AddTimeRangeOption addTimeRangeOption)
{
    m_ranges.add(MediaTime::createWithDouble(start), MediaTime::createWithDouble(end), addTimeRangeOption);
}

bool TimeRanges::contain(double time) const
{
    return m_ranges.contain(MediaTime::createWithDouble(time));
}

size_t TimeRanges::find(double time) const
{
    return m_ranges.find(MediaTime::createWithDouble(time));
}

double TimeRanges::nearest(double time) const
{
    return m_ranges.nearest(MediaTime::createWithDouble(time)).toDouble();
}

double TimeRanges::totalDuration() const
{
    return m_ranges.totalDuration().toDouble();
}

}
