/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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

#include <unicode/uchar.h>
#include <wtf/Expected.h>
#include <wtf/text/StringView.h>

namespace WebCore {

class Decimal;
class QualifiedName;

template<typename CharacterType> bool isComma(CharacterType);
template<typename CharacterType> bool isHTMLSpaceOrComma(CharacterType);
bool isHTMLLineBreak(UChar);
bool isHTMLSpaceButNotLineBreak(UChar);

// 2147483647 is 2^31 - 1.
static const unsigned maxHTMLNonNegativeInteger = 2147483647;

// An implementation of the HTML specification's algorithm to convert a number to a string for number and range types.
String serializeForNumberType(const Decimal&);
String serializeForNumberType(double);

// Convert the specified string to a decimal/double. If the conversion fails, the return value is fallback value or NaN if not specified.
// Leading or trailing illegal characters cause failure, as does passing an empty string.
// The double* parameter may be 0 to check if the string can be parsed without getting the result.
Decimal parseToDecimalForNumberType(StringView);
Decimal parseToDecimalForNumberType(StringView, const Decimal& fallbackValue);
double parseToDoubleForNumberType(StringView);
double parseToDoubleForNumberType(StringView, double fallbackValue);

// http://www.whatwg.org/specs/web-apps/current-work/#rules-for-parsing-integers
enum class HTMLIntegerParsingError { NegativeOverflow, PositiveOverflow, Other };

WEBCORE_EXPORT Expected<int, HTMLIntegerParsingError> parseHTMLInteger(StringView);

// http://www.whatwg.org/specs/web-apps/current-work/#rules-for-parsing-non-negative-integers
WEBCORE_EXPORT Expected<unsigned, HTMLIntegerParsingError> parseHTMLNonNegativeInteger(StringView);

// https://html.spec.whatwg.org/#valid-non-negative-integer
std::optional<int> parseValidHTMLNonNegativeInteger(StringView);

// https://html.spec.whatwg.org/#valid-floating-point-number
std::optional<double> parseValidHTMLFloatingPointNumber(StringView);

// https://html.spec.whatwg.org/#rules-for-parsing-floating-point-number-values
double parseHTMLFloatingPointNumberValue(StringView, double fallbackValue = std::numeric_limits<double>::quiet_NaN());

// https://html.spec.whatwg.org/multipage/infrastructure.html#rules-for-parsing-floating-point-number-values
Vector<double> parseHTMLListOfOfFloatingPointNumberValues(StringView);

// https://html.spec.whatwg.org/multipage/semantics.html#attr-meta-http-equiv-refresh
bool parseMetaHTTPEquivRefresh(StringView, double& delay, String& url);

// https://html.spec.whatwg.org/multipage/infrastructure.html#cors-settings-attribute
String parseCORSSettingsAttribute(const AtomString&);

bool threadSafeMatch(const QualifiedName&, const QualifiedName&);

AtomString parseHTMLHashNameReference(StringView);

// https://html.spec.whatwg.org/#rules-for-parsing-dimension-values
struct HTMLDimension {
    enum class Type : bool { Percentage, Pixel };
    double number;
    Type type;
};
std::optional<HTMLDimension> parseHTMLDimension(StringView);
std::optional<HTMLDimension> parseHTMLMultiLength(StringView);

// Inline implementations of some of the functions declared above.

inline bool isHTMLLineBreak(UChar character)
{
    return character <= '\r' && (character == '\n' || character == '\r');
}

ALWAYS_INLINE bool containsHTMLLineBreak(StringView view)
{
    if (view.is8Bit())
        return charactersContain<LChar, '\r', '\n'>(view.span8());
    return charactersContain<UChar, '\r', '\n'>(view.span16());
}

template<typename CharacterType> inline bool isComma(CharacterType character)
{
    return character == ',';
}

template<typename CharacterType> inline bool isHTMLSpaceOrComma(CharacterType character)
{
    return isComma(character) || isASCIIWhitespace(character);
}

inline bool isHTMLSpaceButNotLineBreak(UChar character)
{
    return isASCIIWhitespace(character) && !isHTMLLineBreak(character);
}

// https://html.spec.whatwg.org/multipage/infrastructure.html#limited-to-only-non-negative-numbers-greater-than-zero
inline unsigned limitToOnlyHTMLNonNegativeNumbersGreaterThanZero(unsigned value, unsigned defaultValue = 1)
{
    return (value > 0 && value <= maxHTMLNonNegativeInteger) ? value : defaultValue;
}

inline unsigned limitToOnlyHTMLNonNegativeNumbersGreaterThanZero(StringView stringValue, unsigned defaultValue = 1)
{
    ASSERT(defaultValue > 0);
    ASSERT(defaultValue <= maxHTMLNonNegativeInteger);
    auto optionalValue = parseHTMLNonNegativeInteger(stringValue);
    unsigned value = optionalValue && optionalValue.value() ? optionalValue.value() : defaultValue;
    ASSERT(value > 0);
    ASSERT(value <= maxHTMLNonNegativeInteger);
    return value;
}

// https://html.spec.whatwg.org/#reflecting-content-attributes-in-idl-attributes:idl-unsigned-long
inline unsigned limitToOnlyHTMLNonNegative(unsigned value, unsigned defaultValue = 0)
{
    ASSERT(defaultValue <= maxHTMLNonNegativeInteger);
    return value <= maxHTMLNonNegativeInteger ? value : defaultValue;
}

inline unsigned limitToOnlyHTMLNonNegative(StringView stringValue, unsigned defaultValue = 0)
{
    ASSERT(defaultValue <= maxHTMLNonNegativeInteger);
    unsigned value = parseHTMLNonNegativeInteger(stringValue).value_or(defaultValue);
    ASSERT(value <= maxHTMLNonNegativeInteger);
    return value;
}

// https://html.spec.whatwg.org/#clamped-to-the-range
inline unsigned clampHTMLNonNegativeIntegerToRange(StringView stringValue, unsigned min, unsigned max, unsigned defaultValue = 0)
{
    ASSERT(defaultValue >= min);
    ASSERT(defaultValue <= max);
    auto optionalValue = parseHTMLNonNegativeInteger(stringValue);
    if (optionalValue)
        return std::min(std::max(optionalValue.value(), min), max);

    return optionalValue.error() == HTMLIntegerParsingError::PositiveOverflow ? max : defaultValue;
}

} // namespace WebCore
