/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

#include <array>
#include <wtf/ASCIICType.h>
#include <wtf/FastFloat.h>
#include <wtf/dtoa/double-conversion.h>
#include <wtf/text/StringView.h>

namespace WTF {

// Only toFixed() can use all the 124 positions. The format is:
// <-> + <21 digits> + decimal point + <100 digits> + null char = 124.
using NumberToStringBuffer = std::array<char, 124>;


// <-> + <320 digits> + decimal point + <6 digits> + null char = 329
using NumberToCSSStringBuffer = std::array<char, 329>;

using NumberToStringSpan = std::span<const char>;

WTF_EXPORT_PRIVATE NumberToStringSpan numberToFixedPrecisionString(float, unsigned significantFigures, NumberToStringBuffer& LIFETIME_BOUND, bool truncateTrailingZeros = false);
WTF_EXPORT_PRIVATE NumberToStringSpan numberToFixedWidthString(float, unsigned decimalPlaces, NumberToStringBuffer& LIFETIME_BOUND);

WTF_EXPORT_PRIVATE NumberToStringSpan numberToStringWithTrailingPoint(double, NumberToStringBuffer& LIFETIME_BOUND);
WTF_EXPORT_PRIVATE NumberToStringSpan numberToFixedPrecisionString(double, unsigned significantFigures, NumberToStringBuffer& LIFETIME_BOUND, bool truncateTrailingZeros = false);
WTF_EXPORT_PRIVATE NumberToStringSpan numberToFixedWidthString(double, unsigned decimalPlaces, NumberToStringBuffer& LIFETIME_BOUND);

WTF_EXPORT_PRIVATE NumberToStringSpan numberToStringAndSize(float, NumberToStringBuffer& LIFETIME_BOUND);
WTF_EXPORT_PRIVATE NumberToStringSpan numberToStringAndSize(double, NumberToStringBuffer& LIFETIME_BOUND);

// Fixed width with up to 6 decimal places, trailing zeros truncated.
WTF_EXPORT_PRIVATE NumberToStringSpan numberToCSSString(double, NumberToCSSStringBuffer& LIFETIME_BOUND);

inline double parseDouble(StringView string, size_t& parsedLength)
{
    if (string.is8Bit())
        return parseDouble(string.span8(), parsedLength);
    return parseDouble(string.span16(), parsedLength);
}

} // namespace WTF

using WTF::NumberToStringBuffer;
using WTF::numberToStringWithTrailingPoint;
using WTF::numberToStringAndSize;
using WTF::numberToFixedPrecisionString;
using WTF::numberToFixedWidthString;
using WTF::parseDouble;
