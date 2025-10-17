/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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

#include <string.h>
#include <time.h>
#include <wtf/DateMath.h>
#include <wtf/Noncopyable.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

class GregorianDateTime final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    GregorianDateTime() = default;
    WTF_EXPORT_PRIVATE explicit GregorianDateTime(double ms, LocalTimeOffset);
    explicit GregorianDateTime(int year, int month, int yearDay, int monthDay, int weekDay, int hour, int minute, int second, int utcOffsetInMinute, bool isDST)
        : m_year(year)
        , m_month(month)
        , m_yearDay(yearDay)
        , m_monthDay(monthDay)
        , m_weekDay(weekDay)
        , m_hour(hour)
        , m_minute(minute)
        , m_second(second)
        , m_utcOffsetInMinute(utcOffsetInMinute)
        , m_isDST(isDST)
    {
    }

    inline int year() const { return m_year; }
    inline int month() const { return m_month; }
    inline int yearDay() const { return m_yearDay; }
    inline int monthDay() const { return m_monthDay; }
    inline int weekDay() const { return m_weekDay; }
    inline int hour() const { return m_hour; }
    inline int minute() const { return m_minute; }
    inline int second() const { return m_second; }
    inline int utcOffsetInMinute() const { return m_utcOffsetInMinute; }
    inline int isDST() const { return m_isDST; }

    inline void setYear(int year) { m_year = year; }
    inline void setMonth(int month) { m_month = month; }
    inline void setYearDay(int yearDay) { m_yearDay = yearDay; }
    inline void setMonthDay(int monthDay) { m_monthDay = monthDay; }
    inline void setWeekDay(int weekDay) { m_weekDay = weekDay; }
    inline void setHour(int hour) { m_hour = hour; }
    inline void setMinute(int minute) { m_minute = minute; }
    inline void setSecond(int second) { m_second = second; }
    inline void setUTCOffsetInMinute(int utcOffsetInMinute) { m_utcOffsetInMinute = utcOffsetInMinute; }
    inline void setIsDST(int isDST) { m_isDST = isDST; }

    static constexpr ptrdiff_t offsetOfYear() { return OBJECT_OFFSETOF(GregorianDateTime, m_year); }
    static constexpr ptrdiff_t offsetOfMonth() { return OBJECT_OFFSETOF(GregorianDateTime, m_month); }
    static constexpr ptrdiff_t offsetOfYearDay() { return OBJECT_OFFSETOF(GregorianDateTime, m_yearDay); }
    static constexpr ptrdiff_t offsetOfMonthDay() { return OBJECT_OFFSETOF(GregorianDateTime, m_monthDay); }
    static constexpr ptrdiff_t offsetOfWeekDay() { return OBJECT_OFFSETOF(GregorianDateTime, m_weekDay); }
    static constexpr ptrdiff_t offsetOfHour() { return OBJECT_OFFSETOF(GregorianDateTime, m_hour); }
    static constexpr ptrdiff_t offsetOfMinute() { return OBJECT_OFFSETOF(GregorianDateTime, m_minute); }
    static constexpr ptrdiff_t offsetOfSecond() { return OBJECT_OFFSETOF(GregorianDateTime, m_second); }
    static constexpr ptrdiff_t offsetOfUTCOffsetInMinute() { return OBJECT_OFFSETOF(GregorianDateTime, m_utcOffsetInMinute); }
    static constexpr ptrdiff_t offsetOfIsDST() { return OBJECT_OFFSETOF(GregorianDateTime, m_isDST); }

    WTF_EXPORT_PRIVATE void setToCurrentLocalTime();

    operator tm() const
    {
        tm ret;
        zeroBytes(ret);

        ret.tm_year = m_year - 1900;
        ret.tm_mon = m_month;
        ret.tm_yday = m_yearDay;
        ret.tm_mday = m_monthDay;
        ret.tm_wday = m_weekDay;
        ret.tm_hour = m_hour;
        ret.tm_min = m_minute;
        ret.tm_sec = m_second;
        ret.tm_isdst = m_isDST;

#if HAVE(TM_GMTOFF)
        ret.tm_gmtoff = static_cast<long>(m_utcOffsetInMinute) * static_cast<long>(secondsPerMinute);
#endif

        return ret;
    }

private:
    int m_year { 0 };
    int m_month { 0 };
    int m_yearDay { 0 };
    int m_monthDay { 0 };
    int m_weekDay { 0 };
    int m_hour { 0 };
    int m_minute { 0 };
    int m_second { 0 };
    int m_utcOffsetInMinute { 0 };
    int m_isDST { 0 };
};

} // namespace WTF

using WTF::GregorianDateTime;
