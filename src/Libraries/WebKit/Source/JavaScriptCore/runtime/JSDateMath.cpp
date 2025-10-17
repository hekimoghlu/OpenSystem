/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#include "JSDateMath.h"

#include "ExceptionHelpers.h"
#include "VM.h"
#include <limits>
#include <wtf/DateMath.h>
#include <wtf/Language.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/unicode/CharacterNames.h>
#include <wtf/unicode/icu/ICUHelpers.h>

#ifdef U_HIDE_DRAFT_API
#undef U_HIDE_DRAFT_API
#endif
#include <unicode/ucal.h>
#define U_HIDE_DRAFT_API 1

namespace JSC {
namespace JSDateMathInternal {
static constexpr bool verbose = false;
}

#if PLATFORM(COCOA)
std::atomic<uint64_t> lastTimeZoneID { 1 };
#endif

class OpaqueICUTimeZone {
    WTF_MAKE_TZONE_ALLOCATED(OpaqueICUTimeZone);
public:
    std::unique_ptr<UCalendar, ICUDeleter<ucal_close>> m_calendar;
    String m_canonicalTimeZoneID;
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(OpaqueICUTimeZone);

void OpaqueICUTimeZoneDeleter::operator()(OpaqueICUTimeZone* timeZone)
{
    if (timeZone)
        delete timeZone;
}

// Get the combined UTC + DST offset for the time passed in.
//
// NOTE: The implementation relies on the fact that no time zones have
// more than one daylight savings offset change per month.
// If this function is called with NaN it returns random value.
LocalTimeOffset DateCache::calculateLocalTimeOffset(double millisecondsFromEpoch, WTF::TimeType inputTimeType)
{
    int32_t rawOffset = 0;
    int32_t dstOffset = 0;
    UErrorCode status = U_ZERO_ERROR;

    // This function can fail input date is invalid: NaN etc.
    // We can return any values in this case since later we fail when computing non timezone offset part anyway.
    constexpr LocalTimeOffset failed { false, 0 };

    auto& timeZoneCache = *this->timeZoneCache();
    ucal_setMillis(timeZoneCache.m_calendar.get(), millisecondsFromEpoch, &status);
    if (U_FAILURE(status))
        return failed;

    if (inputTimeType != WTF::LocalTime) {
        rawOffset = ucal_get(timeZoneCache.m_calendar.get(), UCAL_ZONE_OFFSET, &status);
        if (U_FAILURE(status))
            return failed;
        dstOffset = ucal_get(timeZoneCache.m_calendar.get(), UCAL_DST_OFFSET, &status);
        if (U_FAILURE(status))
            return failed;
    } else {
        ucal_getTimeZoneOffsetFromLocal(timeZoneCache.m_calendar.get(), UCAL_TZ_LOCAL_FORMER, UCAL_TZ_LOCAL_FORMER, &rawOffset, &dstOffset, &status);
        if (U_FAILURE(status))
            return failed;
    }

    return { !!dstOffset, rawOffset + dstOffset };
}

LocalTimeOffsetCache* DateCache::DSTCache::leastRecentlyUsed(LocalTimeOffsetCache* exclude)
{
    LocalTimeOffsetCache* result = nullptr;
    for (auto& cache : m_entries) {
        if (&cache == exclude)
            continue;
        if (!result) {
            result = &cache;
            continue;
        }
        if (result->epoch > cache.epoch)
            result = &cache;
    }
    *result = LocalTimeOffsetCache { };
    return result;
}

std::tuple<LocalTimeOffsetCache*, LocalTimeOffsetCache*> DateCache::DSTCache::probe(int64_t millisecondsFromEpoch)
{
    LocalTimeOffsetCache* before = nullptr;
    LocalTimeOffsetCache* after = nullptr;
    for (auto& cache : m_entries) {
        if (cache.start <= millisecondsFromEpoch) {
            if (!before || before->start < cache.start)
                before = &cache;
        } else if (millisecondsFromEpoch < cache.end) {
            if (!after || after->end > cache.end)
                after = &cache;
        }
    }

    if (!before) {
        if (m_before->isEmpty())
            before = m_before;
        else
            before = leastRecentlyUsed(after);
    }
    if (!after) {
        if (m_after->isEmpty() && before != m_after)
            after = m_after;
        else
            after = leastRecentlyUsed(before);
    }

    m_before = before;
    m_after = after;
    return std::tuple { before, after };
}

void DateCache::DSTCache::extendTheAfterCache(int64_t millisecondsFromEpoch, LocalTimeOffset offset)
{
    if (m_after->offset == offset && m_after->start - defaultDSTDeltaInMilliseconds <= millisecondsFromEpoch && millisecondsFromEpoch <= m_after->end) {
        // Extend the m_after cache.
        m_after->start = millisecondsFromEpoch;
        dataLogLnIf(JSDateMathInternal::verbose, "Cache extended1 from ", millisecondsFromEpoch, " to ", m_after->end, " ", offset.offset, " ", offset.isDST);
    } else {
        // The m_after cache is either invalid or starts too late.
        if (!m_after->isEmpty()) {
            // If the m_after cache is valid, replace it with a new cache.
            m_after = leastRecentlyUsed(m_before);
        }
        m_after->start = millisecondsFromEpoch;
        m_after->end = millisecondsFromEpoch;
        m_after->offset = offset;
        m_after->epoch = bumpEpoch();
        dataLogLnIf(JSDateMathInternal::verbose, "Cache miss, recompute2 ", millisecondsFromEpoch, " ", offset.offset, " ", offset.isDST);
    }
}

LocalTimeOffset DateCache::DSTCache::localTimeOffset(DateCache& dateCache, int64_t millisecondsFromEpoch, WTF::TimeType inputTimeType)
{
    if (millisecondsFromEpoch >= WTF::Int64Milliseconds::minECMAScriptTime && millisecondsFromEpoch <= WTF::Int64Milliseconds::maxECMAScriptTime) {
        // Do nothing. Use millisecondsFromEpoch directly
    } else {
        // Adjust to equivalent time.
        int64_t newTime = WTF::equivalentTime(millisecondsFromEpoch);
        dataLogLnIf(JSDateMathInternal::verbose, "Equivalent time conversion from ", millisecondsFromEpoch, " to ", newTime);
        millisecondsFromEpoch = newTime;
    }

    if (UNLIKELY(m_epoch > UINT32_MAX)) {
        dataLogLnIf(JSDateMathInternal::verbose, "reset DSTCache");
        reset();
    }

    // If the time fits in the cached interval in the last cache hit, return the cached offset.
    if (m_before->start <= millisecondsFromEpoch && millisecondsFromEpoch <= m_before->end) {
        dataLogLnIf(JSDateMathInternal::verbose, "Previous Cache Hit");
        m_before->epoch = bumpEpoch();
        return m_before->offset;
    }

    probe(millisecondsFromEpoch);

    ASSERT(m_before->isEmpty() || m_before->start <= millisecondsFromEpoch);
    ASSERT(m_after->isEmpty() || millisecondsFromEpoch < m_after->start);

    if (m_before->isEmpty()) {
        // Cache miss!
        // Compute the DST offset for the time and shrink the cache interval
        // to only contain the time. This allows fast repeated DST offset
        // computations for the same time.
        LocalTimeOffset offset = dateCache.calculateLocalTimeOffset(millisecondsFromEpoch, inputTimeType);
        m_before->offset = offset;
        m_before->start = millisecondsFromEpoch;
        m_before->end = millisecondsFromEpoch;
        m_before->epoch = bumpEpoch();
        dataLogLnIf(JSDateMathInternal::verbose, "Cache miss, recompute1 ", millisecondsFromEpoch, " ", offset.offset, " ", offset.isDST);
        return offset;
    }

    // Cache hit!
    // If the time fits in the cached interval, return the cached offset.
    if (millisecondsFromEpoch <= m_before->end) {
        dataLogLnIf(JSDateMathInternal::verbose, "Cache hit, simple ", millisecondsFromEpoch, " ", m_before->offset.offset, " ", m_before->offset.isDST);
        m_before->epoch = bumpEpoch();
        return m_before->offset;
    }

    if ((millisecondsFromEpoch - defaultDSTDeltaInMilliseconds) > m_before->end) {
        LocalTimeOffset offset = dateCache.calculateLocalTimeOffset(millisecondsFromEpoch, inputTimeType);
        extendTheAfterCache(millisecondsFromEpoch, offset);
        std::swap(m_before, m_after);
        dataLogLnIf(JSDateMathInternal::verbose, "Cache hit, extend ", millisecondsFromEpoch, " ", offset.offset, " ", offset.isDST);
        return offset;
    }

    m_before->epoch = bumpEpoch();

    // Check if m_after is invalid or starts too late.
    // Note that start of invalid caches is maxECMAScriptTime.
    int64_t newAfterStart = m_before->end < WTF::Int64Milliseconds::maxECMAScriptTime - defaultDSTDeltaInMilliseconds ? m_before->end + defaultDSTDeltaInMilliseconds : WTF::Int64Milliseconds::maxECMAScriptTime;
    if (newAfterStart <= m_after->start) {
        LocalTimeOffset offset = dateCache.calculateLocalTimeOffset(newAfterStart, inputTimeType);
        extendTheAfterCache(newAfterStart, offset);
    } else {
        // Update the usage counter of m_after since it is going to be used.
        ASSERT(!m_after->isEmpty());
        m_after->epoch = bumpEpoch();
    }

    // Now the millisecondsFromEpoch is between m_before->end and m_after->start.
    // Only one daylight savings offset change can occur in this interval.

    if (m_before->offset == m_after->offset) {
        // Merge two caches if they have the same offset.
        m_before->end = m_after->end;
        *m_after = LocalTimeOffsetCache { };
        return m_before->offset;
    }

    // Binary search for daylight savings offset change point,
    // but give up if we don't find it in five iterations.
    for (int i = 4; i >= 0; --i) {
        int64_t delta = m_after->start - m_before->end;
        int64_t middle = !i ? millisecondsFromEpoch : m_before->end + delta / 2;
        LocalTimeOffset offset = dateCache.calculateLocalTimeOffset(middle, inputTimeType);
        if (m_before->offset == offset) {
            m_before->end = middle;
            dataLogLnIf(JSDateMathInternal::verbose, "Cache extended2 from ", m_before->start , " to ", m_before->end, " ", offset.offset, " ", offset.isDST);
            if (millisecondsFromEpoch <= m_before->end)
                return offset;
        } else {
            ASSERT(m_after->offset == offset);
            m_after->start = middle;
            dataLogLnIf(JSDateMathInternal::verbose, "Cache extended3 from ", m_after->start , " to ", m_after->end, " ", offset.offset, " ", offset.isDST);
            if (millisecondsFromEpoch >= m_after->start) {
                // This swap helps the optimistic fast check in subsequent invocations.
                std::swap(m_before, m_after);
                return offset;
            }
        }
    }

    return { };
}

double DateCache::gregorianDateTimeToMS(const GregorianDateTime& t, double milliseconds, WTF::TimeType inputTimeType)
{
    double day = dateToDaysFrom1970(t.year(), t.month(), t.monthDay());
    double ms = timeToMS(t.hour(), t.minute(), t.second(), milliseconds);
    double localTimeResult = (day * WTF::msPerDay) + ms;

    if (inputTimeType == WTF::LocalTime && std::isfinite(localTimeResult))
        return localTimeResult - localTimeOffset(static_cast<int64_t>(localTimeResult), inputTimeType).offset;
    return localTimeResult;
}

double DateCache::localTimeToMS(double milliseconds, WTF::TimeType inputTimeType)
{
    if (inputTimeType == WTF::LocalTime && std::isfinite(milliseconds))
        return milliseconds - localTimeOffset(static_cast<int64_t>(milliseconds), inputTimeType).offset;
    return milliseconds;
}

std::tuple<int32_t, int32_t, int32_t> DateCache::yearMonthDayFromDaysWithCache(int32_t days)
{
    if (m_yearMonthDayCache) {
        // Check conservatively if the given 'days' has
        // the same year and month as the cached 'days'.
        int32_t cachedDays = m_yearMonthDayCache->m_days;
        int32_t newDay = m_yearMonthDayCache->m_day + (days - cachedDays);
        if (newDay >= 1 && newDay <= 28) {
            int32_t year = m_yearMonthDayCache->m_year;
            int32_t month = m_yearMonthDayCache->m_month;
            m_yearMonthDayCache = { days, year, month, newDay };
            return std::tuple { year, month, newDay };
        }
    }
    auto [ year, month, day ] = WTF::yearMonthDayFromDays(days);
    m_yearMonthDayCache = { days, year, month, day };
    return std::tuple { year, month, day };
}

// input is UTC
void DateCache::msToGregorianDateTime(double millisecondsFromEpoch, WTF::TimeType outputTimeType, GregorianDateTime& tm)
{
    LocalTimeOffset localTime;
    if (outputTimeType == WTF::LocalTime && std::isfinite(millisecondsFromEpoch)) {
        localTime = localTimeOffset(static_cast<int64_t>(millisecondsFromEpoch));
        millisecondsFromEpoch += localTime.offset;
    }
    if (std::isfinite(millisecondsFromEpoch)) {
        WTF::Int64Milliseconds timeClipped(static_cast<int64_t>(millisecondsFromEpoch));
        int32_t days = WTF::msToDays(timeClipped);
        int32_t timeInDayMS = WTF::timeInDay(timeClipped, days);
        auto [year, month, day] = yearMonthDayFromDaysWithCache(days);
        int32_t hour = timeInDayMS / (60 * 60 * 1000);
        int32_t minute = (timeInDayMS / (60 * 1000)) % 60;
        int32_t second = (timeInDayMS / 1000) % 60;
        tm = GregorianDateTime(year, month, dayInYear(year, month, day), day, WTF::weekDay(days), hour, minute, second, localTime.offset / WTF::Int64Milliseconds::msPerMinute, localTime.isDST);
    } else
        tm = GregorianDateTime(millisecondsFromEpoch, localTime);
}

double DateCache::parseDate(JSGlobalObject* globalObject, VM& vm, const String& date)
{
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (date == m_cachedDateString)
        return m_cachedDateStringValue;

    // After ICU 72, CLDR generates narrowNoBreakSpace for date time format. Thus, `new Date().toLocaleString('en-US')` starts generating
    // a string including narrowNoBreakSpaces instead of simple spaces. However since code in the wild assumes `new Date(new Date().toLocaleString('en-US'))`
    // works, we need to maintain the ability to parse string including narrowNoBreakSpaces. Rough consensus among implementaters is replacing narrowNoBreakSpaces
    // with simple spaces before parsing.
    String updatedString = makeStringByReplacingAll(date, narrowNoBreakSpace, space);

    auto expectedString = updatedString.tryGetUTF8();
    if (!expectedString) {
        if (expectedString.error() == UTF8ConversionError::OutOfMemory)
            throwOutOfMemoryError(globalObject, scope);
        // https://tc39.github.io/ecma262/#sec-date-objects section 20.3.3.2 states that:
        // "Unrecognizable Strings or dates containing illegal element values in the
        // format String shall cause Date.parse to return NaN."
        return std::numeric_limits<double>::quiet_NaN();
    }

    auto parseDateImpl = [this] (auto dateString) {
        bool isLocalTime;
        double value = WTF::parseES5Date(dateString, isLocalTime);
        if (std::isnan(value))
            value = WTF::parseDate(dateString, isLocalTime);

        if (isLocalTime && std::isfinite(value))
            value -= localTimeOffset(static_cast<int64_t>(value), WTF::LocalTime).offset;

        return value;
    };

    double value = parseDateImpl(expectedString.value().span());
    m_cachedDateString = date;
    m_cachedDateStringValue = value;
    return value;
}

// https://tc39.es/ecma402/#sec-defaulttimezone
String DateCache::defaultTimeZone()
{
    return timeZoneCache()->m_canonicalTimeZoneID;
}

String DateCache::timeZoneDisplayName(bool isDST)
{
    if (m_timeZoneStandardDisplayNameCache.isNull()) {
        auto& timeZoneCache = *this->timeZoneCache();
        CString language = defaultLanguage().utf8();
        {
            Vector<UChar, 32> standardDisplayNameBuffer;
            auto status = callBufferProducingFunction(ucal_getTimeZoneDisplayName, timeZoneCache.m_calendar.get(), UCAL_STANDARD, language.data(), standardDisplayNameBuffer);
            if (U_SUCCESS(status))
                m_timeZoneStandardDisplayNameCache = String::adopt(WTFMove(standardDisplayNameBuffer));
        }
        {
            Vector<UChar, 32> dstDisplayNameBuffer;
            auto status = callBufferProducingFunction(ucal_getTimeZoneDisplayName, timeZoneCache.m_calendar.get(), UCAL_DST, language.data(), dstDisplayNameBuffer);
            if (U_SUCCESS(status))
                m_timeZoneDSTDisplayNameCache = String::adopt(WTFMove(dstDisplayNameBuffer));
        }
    }
    if (isDST)
        return m_timeZoneDSTDisplayNameCache;
    return m_timeZoneStandardDisplayNameCache;
}

static Lock timeZoneCacheLock;

#if PLATFORM(COCOA)
static void timeZoneChangeNotification(CFNotificationCenterRef, void*, CFStringRef, const void*, CFDictionaryRef)
{
    Locker locker { timeZoneCacheLock };
    ASSERT(isMainThread());
    ++lastTimeZoneID;
}
#endif

// To confine icu::TimeZone destructor invocation in this file.
DateCache::DateCache()
{
#if PLATFORM(COCOA)
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        CFNotificationCenterAddObserver(CFNotificationCenterGetLocalCenter(), nullptr, timeZoneChangeNotification, kCFTimeZoneSystemTimeZoneDidChangeNotification, nullptr, CFNotificationSuspensionBehaviorDeliverImmediately);
    });
#endif
}

static std::tuple<String, Vector<UChar, 32>> retrieveTimeZoneInformation()
{
    Locker locker { timeZoneCacheLock };
    static NeverDestroyed<std::tuple<String, Vector<UChar, 32>, uint64_t>> globalCache;

    bool isCacheStale = true;
    uint64_t currentID = 0;
#if PLATFORM(COCOA)
    currentID = lastTimeZoneID.load();
    isCacheStale = std::get<2>(globalCache.get()) != currentID;
#endif
    if (isCacheStale) {
        Vector<UChar, 32> timeZoneID;
        getTimeZoneOverride(timeZoneID);
        String canonical;
        UErrorCode status = U_ZERO_ERROR;
        if (timeZoneID.isEmpty()) {
            status = callBufferProducingFunction(ucal_getHostTimeZone, timeZoneID);
            ASSERT_UNUSED(status, U_SUCCESS(status));
        }
        if (U_SUCCESS(status)) {
            Vector<UChar, 32> canonicalBuffer;
            auto status = callBufferProducingFunction(ucal_getCanonicalTimeZoneID, timeZoneID.data(), timeZoneID.size(), canonicalBuffer, nullptr);
            if (U_SUCCESS(status))
                canonical = String(canonicalBuffer);
        }
        if (canonical.isNull() || isUTCEquivalent(canonical))
            canonical = "UTC"_s;

        globalCache.get() = std::tuple { canonical.isolatedCopy(), WTFMove(timeZoneID), currentID };
    }
    return std::tuple { std::get<0>(globalCache.get()).isolatedCopy(), std::get<1>(globalCache.get()) };
}

DateCache::~DateCache() = default;

Ref<DateInstanceData> DateCache::cachedDateInstanceData(double millisecondsFromEpoch)
{
    return *m_dateInstanceCache.add(millisecondsFromEpoch);
}

void DateCache::timeZoneCacheSlow()
{
    ASSERT(!m_timeZoneCache);
    auto [canonical, timeZoneID] = retrieveTimeZoneInformation();
    auto* cache = new OpaqueICUTimeZone;
    cache->m_canonicalTimeZoneID = WTFMove(canonical);
    UErrorCode status = U_ZERO_ERROR;
    cache->m_calendar = std::unique_ptr<UCalendar, ICUDeleter<ucal_close>>(ucal_open(timeZoneID.data(), timeZoneID.size(), "", UCAL_DEFAULT, &status));
    ASSERT_UNUSED(status, U_SUCCESS(status));
    ucal_setGregorianChange(cache->m_calendar.get(), minECMAScriptTime, &status); // Ignore "unsupported" error.
    m_timeZoneCache = std::unique_ptr<OpaqueICUTimeZone, OpaqueICUTimeZoneDeleter>(cache);
}

void DateCache::resetIfNecessarySlow()
{
    // FIXME: We should clear it only when we know the timezone has been changed on Non-Cocoa platforms.
    // https://bugs.webkit.org/show_bug.cgi?id=218365
    m_timeZoneCache.reset();
    for (auto& cache : m_caches)
        cache.reset();
    m_yearMonthDayCache = std::nullopt;
    m_cachedDateString = String();
    m_cachedDateStringValue = std::numeric_limits<double>::quiet_NaN();
    m_dateInstanceCache.reset();
    m_timeZoneStandardDisplayNameCache = String();
    m_timeZoneDSTDisplayNameCache = String();
}

} // namespace JSC
