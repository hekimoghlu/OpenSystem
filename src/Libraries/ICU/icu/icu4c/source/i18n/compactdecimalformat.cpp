/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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

// Â© 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

// Allow implicit conversion from char16_t* to UnicodeString for this file:
// Helpful in toString methods and elsewhere.
#define UNISTR_FROM_STRING_EXPLICIT

#include "unicode/compactdecimalformat.h"
#include "number_mapper.h"
#include "number_decimfmtprops.h"

using namespace icu;


UOBJECT_DEFINE_RTTI_IMPLEMENTATION(CompactDecimalFormat)


CompactDecimalFormat*
CompactDecimalFormat::createInstance(const Locale& inLocale, UNumberCompactStyle style,
                                     UErrorCode& status) {
    return new CompactDecimalFormat(inLocale, style, status);
}

CompactDecimalFormat::CompactDecimalFormat(const Locale& inLocale, UNumberCompactStyle style,
                                           UErrorCode& status)
        : DecimalFormat(new DecimalFormatSymbols(inLocale, status), status) {
    if (U_FAILURE(status)) return;
    // Minimal properties: let the non-shim code path do most of the logic for us.
    fields->properties.compactStyle = style;
    fields->properties.groupingSize = -2; // do not forward grouping information
    fields->properties.minimumGroupingDigits = 2;
    touch(status);
}

CompactDecimalFormat::CompactDecimalFormat(const CompactDecimalFormat& source) = default;

CompactDecimalFormat::~CompactDecimalFormat() = default;

CompactDecimalFormat& CompactDecimalFormat::operator=(const CompactDecimalFormat& rhs) {
    DecimalFormat::operator=(rhs);
    return *this;
}

CompactDecimalFormat* CompactDecimalFormat::clone() const {
    return new CompactDecimalFormat(*this);
}

void
CompactDecimalFormat::parse(
        const UnicodeString& /* text */,
        Formattable& /* result */,
        ParsePosition& /* parsePosition */) const {
}

void
CompactDecimalFormat::parse(
        const UnicodeString& /* text */,
        Formattable& /* result */,
        UErrorCode& status) const {
    status = U_UNSUPPORTED_ERROR;
}

CurrencyAmount*
CompactDecimalFormat::parseCurrency(
        const UnicodeString& /* text */,
        ParsePosition& /* pos */) const {
    return nullptr;
}


#endif /* #if !UCONFIG_NO_FORMATTING */
