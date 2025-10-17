/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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

#include "JSObject.h"
#include <unicode/udat.h>
#include <wtf/unicode/icu/ICUHelpers.h>

struct UDateIntervalFormat;

namespace JSC {

enum class RelevantExtensionKey : uint8_t;

class JSBoundFunction;

struct UDateIntervalFormatDeleter {
    JS_EXPORT_PRIVATE void operator()(UDateIntervalFormat*);
};

class IntlDateTimeFormat final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlDateTimeFormat*>(cell)->IntlDateTimeFormat::~IntlDateTimeFormat();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlDateTimeFormatSpace<mode>();
    }

    static IntlDateTimeFormat* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    enum class RequiredComponent : uint8_t { Date, Time, Any };
    enum class Defaults : uint8_t { Date, Time, All };
    void initializeDateTimeFormat(JSGlobalObject*, JSValue locales, JSValue options, RequiredComponent, Defaults);
    JSValue format(JSGlobalObject*, double value) const;
    JSValue formatToParts(JSGlobalObject*, double value, JSString* sourceType = nullptr) const;
    JSValue formatRange(JSGlobalObject*, double startDate, double endDate);
    JSValue formatRangeToParts(JSGlobalObject*, double startDate, double endDate);
    JSObject* resolvedOptions(JSGlobalObject*) const;

    JSBoundFunction* boundFormat() const { return m_boundFormat.get(); }
    void setBoundFormat(VM&, JSBoundFunction*);

    static IntlDateTimeFormat* unwrapForOldFunctions(JSGlobalObject*, JSValue);

    enum class HourCycle : uint8_t { None, H11, H12, H23, H24 };
    static HourCycle hourCycleFromPattern(const Vector<UChar, 32>&);

private:
    IntlDateTimeFormat(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    static Vector<String> localeData(const String&, RelevantExtensionKey);

    UDateIntervalFormat* createDateIntervalFormatIfNecessary(JSGlobalObject*);

    static double handleDateTimeValue(JSGlobalObject*, JSValue);

    enum class Weekday : uint8_t { None, Narrow, Short, Long };
    enum class Era : uint8_t { None, Narrow, Short, Long };
    enum class Year : uint8_t { None, TwoDigit, Numeric };
    enum class Month : uint8_t { None, TwoDigit, Numeric, Narrow, Short, Long };
    enum class Day : uint8_t { None, TwoDigit, Numeric };
    enum class DayPeriod : uint8_t { None, Narrow, Short, Long };
    enum class Hour : uint8_t { None, TwoDigit, Numeric };
    enum class Minute : uint8_t { None, TwoDigit, Numeric };
    enum class Second : uint8_t { None, TwoDigit, Numeric };
    enum class TimeZoneName : uint8_t { None, Short, Long, ShortOffset, LongOffset, ShortGeneric, LongGeneric };
    enum class DateTimeStyle : uint8_t { None, Full, Long, Medium, Short };

    void setFormatsFromPattern(StringView);
    static ASCIILiteral hourCycleString(HourCycle);
    static ASCIILiteral weekdayString(Weekday);
    static ASCIILiteral eraString(Era);
    static ASCIILiteral yearString(Year);
    static ASCIILiteral monthString(Month);
    static ASCIILiteral dayString(Day);
    static ASCIILiteral dayPeriodString(DayPeriod);
    static ASCIILiteral hourString(Hour);
    static ASCIILiteral minuteString(Minute);
    static ASCIILiteral secondString(Second);
    static ASCIILiteral timeZoneNameString(TimeZoneName);
    static ASCIILiteral formatStyleString(DateTimeStyle);

    static HourCycle hourCycleFromSymbol(UChar);
    static HourCycle parseHourCycle(const String&);
    static void replaceHourCycleInSkeleton(Vector<UChar, 32>&, bool hour12);
    static void replaceHourCycleInPattern(Vector<UChar, 32>&, HourCycle);
    static String buildSkeleton(Weekday, Era, Year, Month, Day, TriState, HourCycle, Hour, DayPeriod, Minute, Second, unsigned, TimeZoneName);

    using UDateFormatDeleter = ICUDeleter<udat_close>;

    WriteBarrier<JSBoundFunction> m_boundFormat;
    std::unique_ptr<UDateFormat, UDateFormatDeleter> m_dateFormat;
    std::unique_ptr<UDateIntervalFormat, UDateIntervalFormatDeleter> m_dateIntervalFormat;

    String m_locale;
    String m_dataLocale;
    String m_calendar;
    String m_numberingSystem;
    String m_timeZone;
    String m_timeZoneForICU;
    HourCycle m_hourCycle { HourCycle::None };
    Weekday m_weekday { Weekday::None };
    Era m_era { Era::None };
    Year m_year { Year::None };
    Month m_month { Month::None };
    Day m_day { Day::None };
    DayPeriod m_dayPeriod { DayPeriod::None };
    Hour m_hour { Hour::None };
    Minute m_minute { Minute::None };
    Second m_second { Second::None };
    uint8_t m_fractionalSecondDigits { 0 };
    TimeZoneName m_timeZoneName { TimeZoneName::None };
    DateTimeStyle m_dateStyle { DateTimeStyle::None };
    DateTimeStyle m_timeStyle { DateTimeStyle::None };
};

} // namespace JSC
