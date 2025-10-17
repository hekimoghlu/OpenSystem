/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
#include "LocaleICU.h"

#include "LocaleToScriptMapping.h"
#include "LocalizedStrings.h"
#include <limits>
#include <unicode/udatpg.h>
#include <unicode/uloc.h>
#include <unicode/uscript.h>
#include <wtf/DateMath.h>
#include <wtf/text/StringBuffer.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/unicode/icu/ICUHelpers.h>

#if USE(HARFBUZZ)
#include <hb-icu.h>
#include <hb.h>
#endif


namespace WebCore {
using namespace icu;

std::unique_ptr<Locale> Locale::create(const AtomString& locale)
{
    return makeUnique<LocaleICU>(locale.string().utf8().data());
}

LocaleICU::LocaleICU(const char* locale)
    : m_locale(locale)
{
}

LocaleICU::~LocaleICU()
{
#if !UCONFIG_NO_FORMATTING
    unum_close(m_numberFormat);
#endif
    udat_close(m_shortDateFormat);
    udat_close(m_mediumTimeFormat);
    udat_close(m_shortTimeFormat);
}

Locale::WritingDirection LocaleICU::defaultWritingDirection() const
{
#if USE(HARFBUZZ)
    UScriptCode icuScript = localeToScriptCodeForFontSelection(m_locale.span());
    hb_script_t script = hb_icu_script_to_script(icuScript);

    switch (hb_script_get_horizontal_direction(script)) {
    case HB_DIRECTION_LTR:
        return WritingDirection::LeftToRight;
    case HB_DIRECTION_RTL:
        return WritingDirection::RightToLeft;
    default:
        return WritingDirection::Default;
    }
#else
    return WritingDirection::Default;
#endif
}

#if !UCONFIG_NO_FORMATTING
String LocaleICU::decimalSymbol(UNumberFormatSymbol symbol)
{
    UErrorCode status = U_ZERO_ERROR;
    int32_t bufferLength = unum_getSymbol(m_numberFormat, symbol, 0, 0, &status);
    ASSERT(U_SUCCESS(status) || needsToGrowToProduceBuffer(status));
    if (U_FAILURE(status) && !needsToGrowToProduceBuffer(status))
        return String();
    StringBuffer<UChar> buffer(bufferLength);
    status = U_ZERO_ERROR;
    unum_getSymbol(m_numberFormat, symbol, buffer.characters(), bufferLength, &status);
    if (U_FAILURE(status))
        return String();
    return String::adopt(WTFMove(buffer));
}

String LocaleICU::decimalTextAttribute(UNumberFormatTextAttribute tag)
{
    UErrorCode status = U_ZERO_ERROR;
    int32_t bufferLength = unum_getTextAttribute(m_numberFormat, tag, 0, 0, &status);
    ASSERT(U_SUCCESS(status) || needsToGrowToProduceBuffer(status));
    if (U_FAILURE(status) && !needsToGrowToProduceBuffer(status))
        return String();
    StringBuffer<UChar> buffer(bufferLength);
    status = U_ZERO_ERROR;
    unum_getTextAttribute(m_numberFormat, tag, buffer.characters(), bufferLength, &status);
    ASSERT(U_SUCCESS(status));
    if (U_FAILURE(status))
        return String();
    return String::adopt(WTFMove(buffer));
}
#endif

void LocaleICU::initializeLocaleData()
{
#if !UCONFIG_NO_FORMATTING
    if (m_didCreateDecimalFormat)
        return;
    m_didCreateDecimalFormat = true;
    UErrorCode status = U_ZERO_ERROR;
    m_numberFormat = unum_open(UNUM_DECIMAL, 0, 0, m_locale.data(), 0, &status);
    if (!U_SUCCESS(status))
        return;

    Vector<String, DecimalSymbolsSize> symbols;
    symbols.append(decimalSymbol(UNUM_ZERO_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_ONE_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_TWO_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_THREE_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_FOUR_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_FIVE_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_SIX_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_SEVEN_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_EIGHT_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_NINE_DIGIT_SYMBOL));
    symbols.append(decimalSymbol(UNUM_DECIMAL_SEPARATOR_SYMBOL));
    symbols.append(decimalSymbol(UNUM_GROUPING_SEPARATOR_SYMBOL));
    ASSERT(symbols.size() == DecimalSymbolsSize);
    setLocaleData(symbols, decimalTextAttribute(UNUM_POSITIVE_PREFIX), decimalTextAttribute(UNUM_POSITIVE_SUFFIX), decimalTextAttribute(UNUM_NEGATIVE_PREFIX), decimalTextAttribute(UNUM_NEGATIVE_SUFFIX));
#endif
}

bool LocaleICU::initializeShortDateFormat()
{
    if (m_didCreateShortDateFormat)
        return m_shortDateFormat;
    m_shortDateFormat = openDateFormat(UDAT_NONE, UDAT_SHORT);
    m_didCreateShortDateFormat = true;
    return m_shortDateFormat;
}

UDateFormat* LocaleICU::openDateFormat(UDateFormatStyle timeStyle, UDateFormatStyle dateStyle) const
{
    const UChar gmtTimezone[3] = {'G', 'M', 'T'};
    UErrorCode status = U_ZERO_ERROR;
    return udat_open(timeStyle, dateStyle, m_locale.data(), gmtTimezone, std::size(gmtTimezone), 0, -1, &status);
}

static String getDateFormatPattern(const UDateFormat* dateFormat)
{
    if (!dateFormat)
        return emptyString();

    UErrorCode status = U_ZERO_ERROR;
    int32_t length = udat_toPattern(dateFormat, true, 0, 0, &status);
    if (!needsToGrowToProduceBuffer(status) || !length)
        return emptyString();
    StringBuffer<UChar> buffer(length);
    status = U_ZERO_ERROR;
    udat_toPattern(dateFormat, true, buffer.characters(), length, &status);
    if (U_FAILURE(status))
        return emptyString();
    return String::adopt(WTFMove(buffer));
}

std::unique_ptr<Vector<String>> LocaleICU::createLabelVector(const UDateFormat* dateFormat, UDateFormatSymbolType type, int32_t startIndex, int32_t size)
{
    if (!dateFormat)
        return makeUnique<Vector<String>>();
    if (udat_countSymbols(dateFormat, type) != startIndex + size)
        return makeUnique<Vector<String>>();

    auto labels = makeUnique<Vector<String>>();
    labels->reserveInitialCapacity(size);
    for (int32_t i = 0; i < size; ++i) {
        UErrorCode status = U_ZERO_ERROR;
        int32_t length = udat_getSymbols(dateFormat, type, startIndex + i, 0, 0, &status);
        if (!needsToGrowToProduceBuffer(status))
            return makeUnique<Vector<String>>();
        StringBuffer<UChar> buffer(length);
        status = U_ZERO_ERROR;
        udat_getSymbols(dateFormat, type, startIndex + i, buffer.characters(), length, &status);
        if (U_FAILURE(status))
            return makeUnique<Vector<String>>();
        labels->append(String::adopt(WTFMove(buffer)));
    }
    return labels;
}

static std::unique_ptr<Vector<String>> createFallbackMonthLabels()
{
    return makeUnique<Vector<String>>(std::span { WTF::monthFullName });
}

const Vector<String>& LocaleICU::monthLabels()
{
    if (m_monthLabels)
        return *m_monthLabels;
    if (initializeShortDateFormat()) {
        m_monthLabels = createLabelVector(m_shortDateFormat, UDAT_MONTHS, UCAL_JANUARY, 12);
        if (m_monthLabels)
            return *m_monthLabels;
    }
    m_monthLabels = createFallbackMonthLabels();
    return *m_monthLabels;
}

static std::unique_ptr<Vector<String>> createFallbackAMPMLabels()
{
    return makeUnique<Vector<String>>(Vector<String>::from("AM"_str, "PM"_str));
}

void LocaleICU::initializeDateTimeFormat()
{
    if (m_didCreateTimeFormat)
        return;

    // We assume ICU medium time pattern and short time pattern are compatible
    // with LDML, because ICU specific pattern character "V" doesn't appear
    // in both medium and short time pattern.
    m_mediumTimeFormat = openDateFormat(UDAT_MEDIUM, UDAT_NONE);
    m_timeFormatWithSeconds = getDateFormatPattern(m_mediumTimeFormat);

    m_shortTimeFormat = openDateFormat(UDAT_SHORT, UDAT_NONE);
    m_timeFormatWithoutSeconds = getDateFormatPattern(m_shortTimeFormat);

    UDateFormat* dateTimeFormatWithSeconds = openDateFormat(UDAT_MEDIUM, UDAT_SHORT);
    m_dateTimeFormatWithSeconds = getDateFormatPattern(dateTimeFormatWithSeconds);
    udat_close(dateTimeFormatWithSeconds);

    UDateFormat* dateTimeFormatWithoutSeconds = openDateFormat(UDAT_SHORT, UDAT_SHORT);
    m_dateTimeFormatWithoutSeconds = getDateFormatPattern(dateTimeFormatWithoutSeconds);
    udat_close(dateTimeFormatWithoutSeconds);

    auto timeAMPMLabels = createLabelVector(m_mediumTimeFormat, UDAT_AM_PMS, UCAL_AM, 2);
    if (!timeAMPMLabels)
        timeAMPMLabels = createFallbackAMPMLabels();
    m_timeAMPMLabels = *timeAMPMLabels;

    m_didCreateTimeFormat = true;
}

String LocaleICU::dateFormat()
{
    if (!m_dateFormat.isNull())
        return m_dateFormat;
    if (!initializeShortDateFormat())
        return "yyyy-MM-dd"_s;
    m_dateFormat = getDateFormatPattern(m_shortDateFormat);
    return m_dateFormat;
}

static String getFormatForSkeleton(const char* locale, const UChar* skeleton, int32_t skeletonLength)
{
    String format = "yyyy-MM"_s;
    UErrorCode status = U_ZERO_ERROR;
    UDateTimePatternGenerator* patternGenerator = udatpg_open(locale, &status);
    if (!patternGenerator)
        return format;
    status = U_ZERO_ERROR;
    int32_t length = udatpg_getBestPattern(patternGenerator, skeleton, skeletonLength, 0, 0, &status);
    if (needsToGrowToProduceBuffer(status) && length) {
        StringBuffer<UChar> buffer(length);
        status = U_ZERO_ERROR;
        udatpg_getBestPattern(patternGenerator, skeleton, skeletonLength, buffer.characters(), length, &status);
        if (U_SUCCESS(status))
            format = String::adopt(WTFMove(buffer));
    }
    udatpg_close(patternGenerator);
    return format;
}

String LocaleICU::monthFormat()
{
    if (!m_monthFormat.isNull())
        return m_monthFormat;
    // Gets a format for "MMMM" because Windows API always provides formats for
    // "MMMM" in some locales.
    const UChar skeleton[] = { 'y', 'y', 'y', 'y', 'M', 'M', 'M', 'M' };
    m_monthFormat = getFormatForSkeleton(m_locale.data(), skeleton, std::size(skeleton));
    return m_monthFormat;
}

String LocaleICU::shortMonthFormat()
{
    if (!m_shortMonthFormat.isNull())
        return m_shortMonthFormat;
    const UChar skeleton[] = { 'y', 'y', 'y', 'y', 'M', 'M', 'M' };
    m_shortMonthFormat = getFormatForSkeleton(m_locale.data(), skeleton, std::size(skeleton));
    return m_shortMonthFormat;
}

String LocaleICU::timeFormat()
{
    initializeDateTimeFormat();
    return m_timeFormatWithSeconds;
}

String LocaleICU::shortTimeFormat()
{
    initializeDateTimeFormat();
    return m_timeFormatWithoutSeconds;
}

String LocaleICU::dateTimeFormatWithSeconds()
{
    initializeDateTimeFormat();
    return m_dateTimeFormatWithSeconds;
}

String LocaleICU::dateTimeFormatWithoutSeconds()
{
    initializeDateTimeFormat();
    return m_dateTimeFormatWithoutSeconds;
}

const Vector<String>& LocaleICU::shortMonthLabels()
{
    if (!m_shortMonthLabels.isEmpty())
        return m_shortMonthLabels;
    if (initializeShortDateFormat()) {
        if (auto labels = createLabelVector(m_shortDateFormat, UDAT_SHORT_MONTHS, UCAL_JANUARY, 12)) {
            m_shortMonthLabels = *labels;
            return m_shortMonthLabels;
        }
    }
    m_shortMonthLabels.reserveCapacity(std::size(WTF::monthName));
    for (unsigned i = 0; i < std::size(WTF::monthName); ++i)
        m_shortMonthLabels.append(WTF::monthName[i]);
    return m_shortMonthLabels;
}

const Vector<String>& LocaleICU::standAloneMonthLabels()
{
    if (!m_standAloneMonthLabels.isEmpty())
        return m_standAloneMonthLabels;
    if (initializeShortDateFormat()) {
        if (auto labels = createLabelVector(m_shortDateFormat, UDAT_STANDALONE_MONTHS, UCAL_JANUARY, 12)) {
            m_standAloneMonthLabels = *labels;
            return m_standAloneMonthLabels;
        }
    }
    m_standAloneMonthLabels = monthLabels();
    return m_standAloneMonthLabels;
}

const Vector<String>& LocaleICU::shortStandAloneMonthLabels()
{
    if (!m_shortStandAloneMonthLabels.isEmpty())
        return m_shortStandAloneMonthLabels;
    if (initializeShortDateFormat()) {
        if (auto labels = createLabelVector(m_shortDateFormat, UDAT_STANDALONE_SHORT_MONTHS, UCAL_JANUARY, 12)) {
            m_shortStandAloneMonthLabels = *labels;
            return m_shortStandAloneMonthLabels;
        }
    }
    m_shortStandAloneMonthLabels = shortMonthLabels();
    return m_shortStandAloneMonthLabels;
}

const Vector<String>& LocaleICU::timeAMPMLabels()
{
    initializeDateTimeFormat();
    return m_timeAMPMLabels;
}

} // namespace WebCore

