/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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

// Â© 2017 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/numberformatter.h"
#include "number_types.h"

using namespace icu;
using namespace icu::number;
using namespace icu::number::impl;


ScientificNotation Notation::scientific() {
    // NOTE: ISO C++ does not allow C99 designated initializers.
    ScientificSettings settings;
    settings.fEngineeringInterval = 1;
    settings.fRequireMinInt = false;
    settings.fMinExponentDigits = 1;
    settings.fExponentSignDisplay = UNUM_SIGN_AUTO;
    NotationUnion union_;
    union_.scientific = settings;
    return {NTN_SCIENTIFIC, union_};
}

ScientificNotation Notation::engineering() {
    ScientificSettings settings;
    settings.fEngineeringInterval = 3;
    settings.fRequireMinInt = false;
    settings.fMinExponentDigits = 1;
    settings.fExponentSignDisplay = UNUM_SIGN_AUTO;
    NotationUnion union_;
    union_.scientific = settings;
    return {NTN_SCIENTIFIC, union_};
}

ScientificNotation::ScientificNotation(int8_t fEngineeringInterval, bool fRequireMinInt,
                                       impl::digits_t fMinExponentDigits,
                                       UNumberSignDisplay fExponentSignDisplay) {
    ScientificSettings settings;
    settings.fEngineeringInterval = fEngineeringInterval;
    settings.fRequireMinInt = fRequireMinInt;
    settings.fMinExponentDigits = fMinExponentDigits;
    settings.fExponentSignDisplay = fExponentSignDisplay;
    NotationUnion union_;
    union_.scientific = settings;
    *this = {NTN_SCIENTIFIC, union_};
}

Notation Notation::compactShort() {
    NotationUnion union_;
    union_.compactStyle = CompactStyle::UNUM_SHORT;
    return {NTN_COMPACT, union_};
}

Notation Notation::compactLong() {
    NotationUnion union_;
    union_.compactStyle = CompactStyle::UNUM_LONG;
    return {NTN_COMPACT, union_};
}

Notation Notation::simple() {
    return {};
}

ScientificNotation
ScientificNotation::withMinExponentDigits(int32_t minExponentDigits) const {
    if (minExponentDigits >= 1 && minExponentDigits <= kMaxIntFracSig) {
        ScientificSettings settings = fUnion.scientific;
        settings.fMinExponentDigits = static_cast<digits_t>(minExponentDigits);
        NotationUnion union_ = {settings};
        return {NTN_SCIENTIFIC, union_};
    } else {
        return {U_NUMBER_ARG_OUTOFBOUNDS_ERROR};
    }
}

ScientificNotation
ScientificNotation::withExponentSignDisplay(UNumberSignDisplay exponentSignDisplay) const {
    ScientificSettings settings = fUnion.scientific;
    settings.fExponentSignDisplay = exponentSignDisplay;
    NotationUnion union_ = {settings};
    return {NTN_SCIENTIFIC, union_};
}

#endif /* #if !UCONFIG_NO_FORMATTING */
