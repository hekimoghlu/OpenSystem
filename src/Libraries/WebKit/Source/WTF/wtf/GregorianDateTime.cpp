/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
#include <wtf/GregorianDateTime.h>

#include <wtf/DateMath.h>

#if OS(WINDOWS)
#include <windows.h>
#else
#include <time.h>
#endif

namespace WTF {

GregorianDateTime::GregorianDateTime(double ms, LocalTimeOffset localTime)
{
    if (std::isfinite(ms)) {
        WTF::Int64Milliseconds timeClipped(static_cast<int64_t>(ms));
        int32_t days = msToDays(timeClipped);
        int32_t timeInDayMS = timeInDay(timeClipped, days);
        auto [year, month, day] = yearMonthDayFromDays(days);
        int32_t hour = timeInDayMS / (60 * 60 * 1000);
        int32_t minute = (timeInDayMS / (60 * 1000)) % 60;
        int32_t second = (timeInDayMS / 1000) % 60;
        setSecond(second);
        setMinute(minute);
        setHour(hour);
        setWeekDay(WTF::weekDay(days));
        setYearDay(dayInYear(year, month, day));
        setMonthDay(day);
        setMonth(month);
        setYear(year);
    }
    setIsDST(localTime.isDST);
    setUTCOffsetInMinute(localTime.offset / WTF::Int64Milliseconds::msPerMinute);
}

void GregorianDateTime::setToCurrentLocalTime()
{
#if OS(WINDOWS)
    SYSTEMTIME systemTime;
    GetLocalTime(&systemTime);
    TIME_ZONE_INFORMATION timeZoneInformation;
    DWORD timeZoneId = GetTimeZoneInformation(&timeZoneInformation);

    LONG bias = 0;
    if (timeZoneId != TIME_ZONE_ID_INVALID) {
        bias = timeZoneInformation.Bias;
        if (timeZoneId == TIME_ZONE_ID_DAYLIGHT)
            bias += timeZoneInformation.DaylightBias;
        else if ((timeZoneId == TIME_ZONE_ID_STANDARD) || (timeZoneId == TIME_ZONE_ID_UNKNOWN))
            bias += timeZoneInformation.StandardBias;
        else
            ASSERT(0);
    }

    m_year = systemTime.wYear;
    m_month = systemTime.wMonth - 1;
    m_monthDay = systemTime.wDay;
    m_yearDay = dayInYear(m_year, m_month, m_monthDay);
    m_weekDay = systemTime.wDayOfWeek;
    m_hour = systemTime.wHour;
    m_minute = systemTime.wMinute;
    m_second = systemTime.wSecond;
    m_utcOffsetInMinute = -bias;
    m_isDST = timeZoneId == TIME_ZONE_ID_DAYLIGHT ? 1 : 0;
#else
    tm localTM;
    time_t localTime = time(0);
#if HAVE(LOCALTIME_R)
    localtime_r(&localTime, &localTM);
#else
    localtime_s(&localTime, &localTM);
#endif

    m_year = localTM.tm_year + 1900;
    m_month = localTM.tm_mon;
    m_monthDay = localTM.tm_mday;
    m_yearDay = localTM.tm_yday;
    m_weekDay = localTM.tm_wday;
    m_hour = localTM.tm_hour;
    m_minute = localTM.tm_min;
    m_second = localTM.tm_sec;
    m_isDST = localTM.tm_isdst;
#if HAVE(TM_GMTOFF)
    m_utcOffsetInMinute = localTM.tm_gmtoff / secondsPerMinute;
#else
    m_utcOffsetInMinute = calculateLocalTimeOffset(localTime * msPerSecond).offset / msPerMinute;
#endif
#endif
}

} // namespace WTF
