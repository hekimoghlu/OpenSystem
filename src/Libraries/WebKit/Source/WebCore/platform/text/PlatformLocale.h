/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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
#ifndef PlatformLocale_h
#define PlatformLocale_h

#include <wtf/Language.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class DateComponents;

#if PLATFORM(IOS_FAMILY)
class FontCascade;
#endif

class Locale {
    WTF_MAKE_TZONE_ALLOCATED(Locale);
    WTF_MAKE_NONCOPYABLE(Locale);

public:
    WEBCORE_EXPORT static std::unique_ptr<Locale> create(const AtomString& localeIdentifier);
    static std::unique_ptr<Locale> createDefault();

    // Converts the specified number string to another number string localized
    // for this Locale locale. The input string must conform to HTML
    // floating-point numbers, and is not empty.
    String convertToLocalizedNumber(const String&);

    // Converts the specified localized number string to a number string in the
    // HTML floating-point number format. The input string is provided by a end
    // user, and might not be a number string. It's ok that the function returns
    // a string which is not conforms to the HTML floating-point number format,
    // callers of this function are responsible to check the format of the
    // resultant string.
    String convertFromLocalizedNumber(const String&);

    enum class WritingDirection: uint8_t {
        Default,
        LeftToRight,
        RightToLeft
    };

    // Returns the default writing direction for the specified locale.
    virtual WritingDirection defaultWritingDirection() const;

    // Returns date format in Unicode TR35 LDML[1] containing day of month,
    // month, and year, e.g. "dd/mm/yyyy"
    // [1] LDML http://unicode.org/reports/tr35/#Date_Format_Patterns
    virtual String dateFormat() = 0;

    // Returns a year-month format in Unicode TR35 LDML.
    virtual String monthFormat() = 0;

    // Returns a year-month format using short month lanel in Unicode TR35 LDML.
    virtual String shortMonthFormat() = 0;

    // Returns time format in Unicode TR35 LDML[1] containing hour, minute, and
    // second with optional period(AM/PM), e.g. "h:mm:ss a"
    // [1] LDML http://unicode.org/reports/tr35/#Date_Format_Patterns
    virtual String timeFormat() = 0;

    // Returns time format in Unicode TR35 LDML containing hour, and minute
    // with optional period(AM/PM), e.g. "h:mm a"
    // Note: Some platforms return same value as timeFormat().
    virtual String shortTimeFormat() = 0;

    // Returns a date-time format in Unicode TR35 LDML. It should have a seconds
    // field.
    virtual String dateTimeFormatWithSeconds() = 0;

    // Returns a date-time format in Unicode TR35 LDML. It should have no seconds
    // field.
    virtual String dateTimeFormatWithoutSeconds() = 0;

    // Returns a vector of string of which size is 12. The first item is a
    // localized string of Jan and the last item is a localized string of
    // Dec. These strings should be short.
    virtual const Vector<String>& shortMonthLabels() = 0;

    // Returns a vector of string of which size is 12. The first item is a
    // stand-alone localized string of January and the last item is a
    // stand-alone localized string of December. These strings should not be
    // abbreviations.
    virtual const Vector<String>& standAloneMonthLabels() = 0;

    // Stand-alone month version of shortMonthLabels.
    virtual const Vector<String>& shortStandAloneMonthLabels() = 0;

    // Returns localized period field(AM/PM) strings.
    virtual const Vector<String>& timeAMPMLabels() = 0;

    // Returns a vector of string of which size is 12. The first item is a
    // localized string of January, and the last item is a localized string of
    // December. These strings should not be abbreviations.
    virtual const Vector<String>& monthLabels() = 0;

    String localizedDecimalSeparator();

    enum FormatType { FormatTypeUnspecified, FormatTypeShort, FormatTypeMedium };

    // Serializes the specified date into a formatted date string to
    // display to the user. If an implementation doesn't support
    // localized dates the function should return an empty string.
    // FormatType can be used to specify if you want the short format. 
    virtual String formatDateTime(const DateComponents&, FormatType = FormatTypeUnspecified);

    WEBCORE_EXPORT virtual ~Locale();

protected:
    enum {
        // 0-9 for digits.
        DecimalSeparatorIndex = 10,
        GroupSeparatorIndex = 11,
        DecimalSymbolsSize
    };

    Locale() : m_hasLocaleData(false) { }
    virtual void initializeLocaleData() = 0;
    void setLocaleData(const Vector<String, DecimalSymbolsSize>&, const String& positivePrefix, const String& positiveSuffix, const String& negativePrefix, const String& negativeSuffix);

private:
    bool detectSignAndGetDigitRange(const String& input, bool& isNegative, unsigned& startIndex, unsigned& endIndex);
    unsigned matchedDecimalSymbolIndex(const String& input, unsigned& position);

    std::array<String, DecimalSymbolsSize> m_decimalSymbols;
    String m_positivePrefix;
    String m_positiveSuffix;
    String m_negativePrefix;
    String m_negativeSuffix;
    bool m_hasLocaleData;
};

inline std::unique_ptr<Locale> Locale::createDefault()
{
    return Locale::create(AtomString { defaultLanguage() });
}

}
#endif
