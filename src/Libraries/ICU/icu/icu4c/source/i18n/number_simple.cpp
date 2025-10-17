/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#include "unicode/simplenumberformatter.h"
#include "number_formatimpl.h"
#include "number_utils.h"
#include "number_patternmodifier.h"
#include "number_utypes.h"

using namespace icu;
using namespace icu::number;
using namespace icu::number::impl;


SimpleNumber
SimpleNumber::forInt64(int64_t value, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return {};
    }
    auto* results = new UFormattedNumberData();
    if (results == nullptr) {
        status = U_MEMORY_ALLOCATION_ERROR;
        return {};
    }
    results->quantity.setToLong(value);
    return SimpleNumber(results, status);
}

SimpleNumber::SimpleNumber(UFormattedNumberData* data, UErrorCode& status) : fData(data) {
    if (U_FAILURE(status)) {
        return;
    }
    if (fData == nullptr) {
        status = U_ILLEGAL_ARGUMENT_ERROR;
        return;
    }
    if (fData->quantity.isNegative()) {
        fSign = UNUM_SIMPLE_NUMBER_MINUS_SIGN;
    } else {
        fSign = UNUM_SIMPLE_NUMBER_NO_SIGN;
    }
}

void SimpleNumber::cleanup() {
    delete fData;
    fData = nullptr;
}

void SimpleNumber::multiplyByPowerOfTen(int32_t power, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return;
    }
    if (fData == nullptr) {
        status = U_INVALID_STATE_ERROR;
        return;
    }
    fData->quantity.adjustMagnitude(power);
}

void SimpleNumber::roundTo(int32_t position, UNumberFormatRoundingMode roundingMode, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return;
    }
    if (fData == nullptr) {
        status = U_INVALID_STATE_ERROR;
        return;
    }
    fData->quantity.roundToMagnitude(position, roundingMode, status);
}

void SimpleNumber::setMinimumIntegerDigits(uint32_t position, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return;
    }
    if (fData == nullptr) {
        status = U_INVALID_STATE_ERROR;
        return;
    }
    fData->quantity.decreaseMinIntegerTo(position);
    fData->quantity.increaseMinIntegerTo(position);
}

void SimpleNumber::setMinimumFractionDigits(uint32_t position, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return;
    }
    if (fData == nullptr) {
        status = U_INVALID_STATE_ERROR;
        return;
    }
    fData->quantity.setMinFraction(position);
}

void SimpleNumber::setMaximumIntegerDigits(uint32_t position, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return;
    }
    if (fData == nullptr) {
        status = U_INVALID_STATE_ERROR;
        return;
    }
    fData->quantity.decreaseMinIntegerTo(position);
    fData->quantity.applyMaxInteger(position);
}

void SimpleNumber::setSign(USimpleNumberSign sign, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return;
    }
    if (fData == nullptr) {
        status = U_INVALID_STATE_ERROR;
        return;
    }
    fSign = sign;
}


void SimpleNumberFormatter::cleanup() {
    delete fOwnedSymbols;
    delete fMicros;
    delete fPatternModifier;
    fOwnedSymbols = nullptr;
    fMicros = nullptr;
    fPatternModifier = nullptr;
}

SimpleNumberFormatter SimpleNumberFormatter::forLocale(const icu::Locale &locale, UErrorCode &status) {
    return SimpleNumberFormatter::forLocaleAndGroupingStrategy(locale, UNUM_GROUPING_AUTO, status);
}

SimpleNumberFormatter SimpleNumberFormatter::forLocaleAndGroupingStrategy(
        const icu::Locale &locale,
        UNumberGroupingStrategy groupingStrategy,
        UErrorCode &status) {
    SimpleNumberFormatter retval;
    retval.fOwnedSymbols = new DecimalFormatSymbols(locale, status);
    if (U_FAILURE(status)) {
        return retval;
    }
    if (retval.fOwnedSymbols == nullptr) {
        status = U_MEMORY_ALLOCATION_ERROR;
        return retval;
    }
    retval.initialize(locale, *retval.fOwnedSymbols, groupingStrategy, status);
    return retval;
}


SimpleNumberFormatter SimpleNumberFormatter::forLocaleAndSymbolsAndGroupingStrategy(
        const icu::Locale &locale,
        const DecimalFormatSymbols &symbols,
        UNumberGroupingStrategy groupingStrategy,
        UErrorCode &status) {
    SimpleNumberFormatter retval;
    retval.initialize(locale, symbols, groupingStrategy, status);
    return retval;
}


void SimpleNumberFormatter::initialize(
        const icu::Locale &locale,
        const DecimalFormatSymbols &symbols,
        UNumberGroupingStrategy groupingStrategy,
        UErrorCode &status) {
    if (U_FAILURE(status)) {
        return;
    }

    fMicros = new SimpleMicroProps();
    if (fMicros == nullptr) {
        status = U_MEMORY_ALLOCATION_ERROR;
        return;
    }
    fMicros->symbols = &symbols;

    const auto* pattern = utils::getPatternForStyle(
        locale,
        symbols.getNumberingSystemName(),
        CLDR_PATTERN_STYLE_DECIMAL,
        status);
    if (U_FAILURE(status)) {
        return;
    }

    ParsedPatternInfo patternInfo;
    PatternParser::parseToPatternInfo(UnicodeString(pattern), patternInfo, status);
    if (U_FAILURE(status)) {
        return;
    }

    auto grouper = Grouper::forStrategy(groupingStrategy);
    grouper.setLocaleData(patternInfo, locale);
    fMicros->grouping = grouper;

    MutablePatternModifier patternModifier(false);
    patternModifier.setPatternInfo(&patternInfo, kUndefinedField);
    patternModifier.setPatternAttributes(UNUM_SIGN_EXCEPT_ZERO, false, false);
    patternModifier.setSymbols(fMicros->symbols, {}, UNUM_UNIT_WIDTH_SHORT, nullptr, status);

    fPatternModifier = new AdoptingSignumModifierStore(patternModifier.createImmutableForPlural(StandardPlural::COUNT, status));

    fGroupingStrategy = groupingStrategy;
}

FormattedNumber SimpleNumberFormatter::format(SimpleNumber value, UErrorCode &status) const {
    formatImpl(value.fData, value.fSign, status);

    // Do not save the results object if we encountered a failure.
    if (U_SUCCESS(status)) {
        auto* temp = value.fData;
        value.fData = nullptr;
        return FormattedNumber(temp);
    } else {
        return FormattedNumber(status);
    }
}

void SimpleNumberFormatter::formatImpl(UFormattedNumberData* data, USimpleNumberSign sign, UErrorCode &status) const {
    if (U_FAILURE(status)) {
        return;
    }
    if (data == nullptr) {
        status = U_ILLEGAL_ARGUMENT_ERROR;
        return;
    }
    if (fPatternModifier == nullptr || fMicros == nullptr) {
        status = U_INVALID_STATE_ERROR;
        return;
    }

    Signum signum;
    if (sign == UNUM_SIMPLE_NUMBER_MINUS_SIGN) {
        signum = SIGNUM_NEG;
    } else if (sign == UNUM_SIMPLE_NUMBER_PLUS_SIGN) {
        signum = SIGNUM_POS;
    } else {
        signum = SIGNUM_POS_ZERO;
    }

    const Modifier* modifier = (*fPatternModifier)[signum];
    auto length = NumberFormatterImpl::writeNumber(
        *fMicros,
        data->quantity,
        data->getStringRef(),
        0,
        status);
    length += modifier->apply(data->getStringRef(), 0, length, status);
    data->getStringRef().writeTerminator(status);
}

#endif /* #if !UCONFIG_NO_FORMATTING */
