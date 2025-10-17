/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#include <wtf/dtoa.h>
#include <wtf/dragonbox/dragonbox_to_chars.h>

namespace WTF {

NumberToStringSpan numberToStringAndSize(float number, NumberToStringBuffer& buffer)
{
    static_assert(sizeof(buffer) >= (dragonbox::max_string_length<dragonbox::ieee754_binary32>() + 1));
    auto* result = dragonbox::detail::to_chars_n<WTF::dragonbox::Mode::ToShortest>(number, buffer.data());
    return std::span { buffer }.first(result - buffer.data());
}

NumberToStringSpan numberToStringAndSize(double number, NumberToStringBuffer& buffer)
{
    static_assert(sizeof(buffer) >= (dragonbox::max_string_length<dragonbox::ieee754_binary64>() + 1));
    auto* result = dragonbox::detail::to_chars_n<WTF::dragonbox::Mode::ToShortest>(number, buffer.data());
    return std::span { buffer }.first(result - buffer.data());
}

NumberToStringSpan numberToStringWithTrailingPoint(double d, NumberToStringBuffer& buffer)
{
    double_conversion::StringBuilder builder(std::span<char> { buffer });
    auto& converter = double_conversion::DoubleToStringConverter::EcmaScriptConverterWithTrailingPoint();
    converter.ToShortest(d, &builder);
    return builder.Finalize();
}

static inline void truncateTrailingZeros(std::span<const char> buffer, double_conversion::StringBuilder& builder)
{
    size_t length = builder.position();
    size_t decimalPointPosition = 0;
    for (; decimalPointPosition < length; ++decimalPointPosition) {
        if (buffer[decimalPointPosition] == '.')
            break;
    }

    // No decimal separator found, early exit.
    if (decimalPointPosition == length)
        return;

    size_t pastMantissa = decimalPointPosition + 1;
    for (; pastMantissa < length; ++pastMantissa) {
        if (buffer[pastMantissa] == 'e')
            break;
    }

    size_t truncatedLength = pastMantissa;
    for (; truncatedLength > decimalPointPosition + 1; --truncatedLength) {
        if (buffer[truncatedLength - 1] != '0')
            break;
    }

    // No trailing zeros found to strip.
    if (truncatedLength == pastMantissa)
        return;

    // If we removed all trailing zeros, remove the decimal point as well.
    if (truncatedLength == decimalPointPosition + 1)
        truncatedLength = decimalPointPosition;

    // Truncate the mantissa, and return the final result.
    builder.RemoveCharacters(truncatedLength, pastMantissa);
}

NumberToStringSpan numberToFixedPrecisionString(float number, unsigned significantFigures, NumberToStringBuffer& buffer, bool shouldTruncateTrailingZeros)
{
    // For now, just call the double precision version.
    // Do that here instead of at callers to pave the way to add a more efficient code path later.
    return numberToFixedPrecisionString(static_cast<double>(number), significantFigures, buffer, shouldTruncateTrailingZeros);
}

NumberToStringSpan numberToFixedPrecisionString(double d, unsigned significantFigures, NumberToStringBuffer& buffer, bool shouldTruncateTrailingZeros)
{
    // Mimic sprintf("%.[precision]g", ...).
    // "g": Signed value printed in f or e format, whichever is more compact for the given value and precision.
    // The e format is used only when the exponent of the value is less than â€“4 or greater than or equal to the
    // precision argument. Trailing zeros are truncated, and the decimal point appears only if one or more digits follow it.
    // "precision": The precision specifies the maximum number of significant digits printed.
    double_conversion::StringBuilder builder(std::span<char> { buffer });
    auto& converter = double_conversion::DoubleToStringConverter::EcmaScriptConverter();
    converter.ToPrecision(d, significantFigures, &builder);
    if (shouldTruncateTrailingZeros)
        truncateTrailingZeros(std::span<const char> { buffer }, builder);
    return builder.Finalize();
}

NumberToStringSpan numberToFixedWidthString(float number, unsigned decimalPlaces, NumberToStringBuffer& buffer)
{
    // For now, just call the double precision version.
    // Do that here instead of at callers to pave the way to add a more efficient code path later.
    return numberToFixedWidthString(static_cast<double>(number), decimalPlaces, buffer);
}

NumberToStringSpan numberToFixedWidthString(double d, unsigned decimalPlaces, NumberToStringBuffer& buffer)
{
    // Mimic sprintf("%.[precision]f", ...).
    // "f": Signed value having the form [ â€“ ]dddd.dddd, where dddd is one or more decimal digits.
    // The number of digits before the decimal point depends on the magnitude of the number, and
    // the number of digits after the decimal point depends on the requested precision.
    // "precision": The precision value specifies the number of digits after the decimal point.
    // If a decimal point appears, at least one digit appears before it.
    // The value is rounded to the appropriate number of digits.    
    double_conversion::StringBuilder builder(std::span<char> { buffer });
    auto& converter = double_conversion::DoubleToStringConverter::EcmaScriptConverter();
    converter.ToFixed(d, decimalPlaces, &builder);
    return builder.Finalize();
}

NumberToStringSpan numberToCSSString(double d, NumberToCSSStringBuffer& buffer)
{
    // Mimic sprintf("%.[precision]f", ...).
    // "f": Signed value having the form [ â€“ ]dddd.dddd, where dddd is one or more decimal digits.
    // The number of digits before the decimal point depends on the magnitude of the number, and
    // the number of digits after the decimal point depends on the requested precision.
    // "precision": The precision value specifies the number of digits after the decimal point.
    // If a decimal point appears, at least one digit appears before it.
    // The value is rounded to the appropriate number of digits.
    double_conversion::StringBuilder builder(std::span<char> { buffer });
    auto& converter = double_conversion::DoubleToStringConverter::CSSConverter();
    converter.ToFixedUncapped(d, 6, &builder);
    truncateTrailingZeros(std::span<const char> { buffer }, builder);
    // If we've truncated the trailing zeros and a trailing decimal, we may have a -0. Remove the negative sign in this case.
    if (builder.position() == 2 && buffer[0] == '-' && buffer[1] == '0')
        builder.RemoveCharacters(0, 1);
    return builder.Finalize();
}

} // namespace WTF
