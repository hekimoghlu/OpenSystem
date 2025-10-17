/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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

#include "numparse_types.h"
#include "number_decimalquantity.h"
#include "string_segment.h"
#include "putilimp.h"
#include <cmath>

using namespace icu;
using namespace icu::number;
using namespace icu::number::impl;
using namespace icu::numparse;
using namespace icu::numparse::impl;


ParsedNumber::ParsedNumber() {
    clear();
}

void ParsedNumber::clear() {
    quantity.bogus = true;
    charEnd = 0;
    flags = 0;
    prefix.setToBogus();
    suffix.setToBogus();
    currencyCode[0] = 0;
}

void ParsedNumber::setCharsConsumed(const StringSegment& segment) {
    charEnd = segment.getOffset();
}

void ParsedNumber::postProcess() {
    if (!quantity.bogus && 0 != (flags & FLAG_NEGATIVE)) {
        quantity.negate();
    }
}

bool ParsedNumber::success() const {
    return charEnd > 0 && 0 == (flags & FLAG_FAIL);
}

bool ParsedNumber::seenNumber() const {
    return !quantity.bogus || 0 != (flags & FLAG_NAN) || 0 != (flags & FLAG_INFINITY);
}

double ParsedNumber::getDouble(UErrorCode& status) const {
    bool sawNaN = 0 != (flags & FLAG_NAN);
    bool sawInfinity = 0 != (flags & FLAG_INFINITY);

    // Check for NaN, infinity, and -0.0
    if (sawNaN) {
        // Can't use NAN or std::nan because the byte pattern is platform-dependent;
        // MSVC sets the sign bit, but Clang and GCC do not
        return uprv_getNaN();
    }
    if (sawInfinity) {
        if (0 != (flags & FLAG_NEGATIVE)) {
            return -INFINITY;
        } else {
            return INFINITY;
        }
    }
    if (quantity.bogus) {
        status = U_INVALID_STATE_ERROR;
        return 0.0;
    }
    if (quantity.isZeroish() && quantity.isNegative()) {
        return -0.0;
    }

    if (quantity.fitsInLong()) {
        return static_cast<double>(quantity.toLong());
    } else {
        return quantity.toDouble();
    }
}

void ParsedNumber::populateFormattable(Formattable& output, parse_flags_t parseFlags) const {
    bool sawNaN = 0 != (flags & FLAG_NAN);
    bool sawInfinity = 0 != (flags & FLAG_INFINITY);
    bool integerOnly = 0 != (parseFlags & PARSE_FLAG_INTEGER_ONLY);

    // Check for NaN, infinity, and -0.0
    if (sawNaN) {
        // Can't use NAN or std::nan because the byte pattern is platform-dependent;
        // MSVC sets the sign bit, but Clang and GCC do not
        output.setDouble(uprv_getNaN());
        return;
    }
    if (sawInfinity) {
        if (0 != (flags & FLAG_NEGATIVE)) {
            output.setDouble(-INFINITY);
            return;
        } else {
            output.setDouble(INFINITY);
            return;
        }
    }
    U_ASSERT(!quantity.bogus);
    if (quantity.isZeroish() && quantity.isNegative() && !integerOnly) {
        output.setDouble(-0.0);
        return;
    }

    // All other numbers
    output.adoptDecimalQuantity(new DecimalQuantity(quantity));
}

bool ParsedNumber::isBetterThan(const ParsedNumber& other) {
    // Favor results with strictly more characters consumed.
    return charEnd > other.charEnd;
}



#endif /* #if !UCONFIG_NO_FORMATTING */
