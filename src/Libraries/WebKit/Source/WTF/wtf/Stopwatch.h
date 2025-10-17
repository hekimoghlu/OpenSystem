/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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

#include <cmath>
#include <utility>
#include <wtf/MonotonicTime.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WTF {

class Stopwatch final : public RefCounted<Stopwatch> {
public:
    static Ref<Stopwatch> create()
    {
        return adoptRef(*new Stopwatch());
    }

    void reset();
    void start();
    void stop();

    Seconds elapsedTime() const;
    Seconds elapsedTimeSince(MonotonicTime) const;
    std::tuple<Seconds, MonotonicTime> elapsedTimeAndTimestamp() const;

    std::optional<Seconds> fromMonotonicTime(MonotonicTime) const;

    bool isActive() const { return !m_lastStartTime.isNaN(); }
private:
    Stopwatch() { reset(); }

    Seconds m_elapsedTime;
    MonotonicTime m_lastStartTime;
    Vector<std::pair<MonotonicTime, MonotonicTime>> m_pastInternals;
};

inline void Stopwatch::reset()
{
    m_elapsedTime = 0_s;
    m_lastStartTime = MonotonicTime::nan();
}

inline void Stopwatch::start()
{
    ASSERT_WITH_MESSAGE(m_lastStartTime.isNaN(), "Tried to start the stopwatch, but it is already running.");

    m_lastStartTime = MonotonicTime::now();
}

inline void Stopwatch::stop()
{
    ASSERT_WITH_MESSAGE(!m_lastStartTime.isNaN(), "Tried to stop the stopwatch, but it is not running.");

    auto stopTime = MonotonicTime::now();
    m_pastInternals.append({ m_lastStartTime, stopTime });
    m_elapsedTime += stopTime - m_lastStartTime;
    m_lastStartTime = MonotonicTime::nan();
}

inline Seconds Stopwatch::elapsedTime() const
{
    return std::get<0>(elapsedTimeAndTimestamp());
}

inline std::tuple<Seconds, MonotonicTime> Stopwatch::elapsedTimeAndTimestamp() const
{
    auto timestamp = MonotonicTime::now();
    if (!isActive())
        return std::tuple { m_elapsedTime, timestamp };

    return std::tuple { m_elapsedTime + (timestamp - m_lastStartTime), timestamp };
}

inline Seconds Stopwatch::elapsedTimeSince(MonotonicTime timeStamp) const
{
    if (!isActive())
        return m_elapsedTime;

    return m_elapsedTime + (timeStamp - m_lastStartTime);
}

inline std::optional<Seconds> Stopwatch::fromMonotonicTime(MonotonicTime timeStamp) const
{
    if (!m_lastStartTime.isNaN() && m_lastStartTime < timeStamp)
        return Stopwatch::elapsedTimeSince(timeStamp);

    Seconds elapsedTime;
    for (auto& interval : m_pastInternals) {
        if (timeStamp < interval.first)
            return std::nullopt;
        if (interval.first <= timeStamp && timeStamp <= interval.second)
            return elapsedTime + timeStamp - interval.first;
        elapsedTime += interval.second - interval.first;
    }

    return std::nullopt;
}

} // namespace WTF

using WTF::Stopwatch;
