/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

#include <stdlib.h>
#include <cmath>
#include "number_asformat.h"
#include "number_types.h"
#include "number_utils.h"
#include "fphdlimp.h"
#include "number_utypes.h"

using namespace icu;
using namespace icu::number;
using namespace icu::number::impl;

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(LocalizedNumberFormatterAsFormat)

LocalizedNumberFormatterAsFormat::LocalizedNumberFormatterAsFormat(
        const LocalizedNumberFormatter& formatter, const Locale& locale)
        : fFormatter(formatter), fLocale(locale) {
    const char* localeName = locale.getName();
    setLocaleIDs(localeName, localeName);
}

LocalizedNumberFormatterAsFormat::~LocalizedNumberFormatterAsFormat() = default;

bool LocalizedNumberFormatterAsFormat::operator==(const Format& other) const {
    const auto* _other = dynamic_cast<const LocalizedNumberFormatterAsFormat*>(&other);
    if (_other == nullptr) {
        return false;
    }
    // TODO: Change this to use LocalizedNumberFormatter::operator== if it is ever proposed.
    // This implementation is fine, but not particularly efficient.
    UErrorCode localStatus = U_ZERO_ERROR;
    return fFormatter.toSkeleton(localStatus) == _other->fFormatter.toSkeleton(localStatus);
}

LocalizedNumberFormatterAsFormat* LocalizedNumberFormatterAsFormat::clone() const {
    return new LocalizedNumberFormatterAsFormat(*this);
}

UnicodeString& LocalizedNumberFormatterAsFormat::format(const Formattable& obj, UnicodeString& appendTo,
                                                        FieldPosition& pos, UErrorCode& status) const {
    if (U_FAILURE(status)) { return appendTo; }
    UFormattedNumberData data;
    obj.populateDecimalQuantity(data.quantity, status);
    if (U_FAILURE(status)) {
        return appendTo;
    }
    fFormatter.formatImpl(&data, status);
    if (U_FAILURE(status)) {
        return appendTo;
    }
    // always return first occurrence:
    pos.setBeginIndex(0);
    pos.setEndIndex(0);
    bool found = data.nextFieldPosition(pos, status);
    if (found && appendTo.length() != 0) {
        pos.setBeginIndex(pos.getBeginIndex() + appendTo.length());
        pos.setEndIndex(pos.getEndIndex() + appendTo.length());
    }
    appendTo.append(data.toTempString(status));
    return appendTo;
}

UnicodeString& LocalizedNumberFormatterAsFormat::format(const Formattable& obj, UnicodeString& appendTo,
                                                        FieldPositionIterator* posIter,
                                                        UErrorCode& status) const {
    if (U_FAILURE(status)) { return appendTo; }
    UFormattedNumberData data;
    obj.populateDecimalQuantity(data.quantity, status);
    if (U_FAILURE(status)) {
        return appendTo;
    }
    fFormatter.formatImpl(&data, status);
    if (U_FAILURE(status)) {
        return appendTo;
    }
    appendTo.append(data.toTempString(status));
    if (posIter != nullptr) {
        FieldPositionIteratorHandler fpih(posIter, status);
        data.getAllFieldPositions(fpih, status);
    }
    return appendTo;
}

void LocalizedNumberFormatterAsFormat::parseObject(const UnicodeString&, Formattable&,
                                                   ParsePosition& parse_pos) const {
    // Not supported.
    parse_pos.setErrorIndex(0);
}

const LocalizedNumberFormatter& LocalizedNumberFormatterAsFormat::getNumberFormatter() const {
    return fFormatter;
}


// Definitions of public API methods (put here for dependency disentanglement)

Format* LocalizedNumberFormatter::toFormat(UErrorCode& status) const {
    if (U_FAILURE(status)) {
        return nullptr;
    }
    LocalPointer<LocalizedNumberFormatterAsFormat> retval(
            new LocalizedNumberFormatterAsFormat(*this, fMacros.locale), status);
    return retval.orphan();
}

#endif /* #if !UCONFIG_NO_FORMATTING */
