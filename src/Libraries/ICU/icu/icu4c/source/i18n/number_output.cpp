/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

// Â© 2019 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/measunit.h"
#include "unicode/numberformatter.h"
#include "number_utypes.h"
#include "util.h"
#include "number_decimalquantity.h"
#include "number_decnum.h"
#include "numrange_impl.h"

U_NAMESPACE_BEGIN
namespace number {


UPRV_FORMATTED_VALUE_SUBCLASS_AUTO_IMPL(FormattedNumber)

#define UPRV_NOARG

void FormattedNumber::toDecimalNumber(ByteSink& sink, UErrorCode& status) const {
    UPRV_FORMATTED_VALUE_METHOD_GUARD(UPRV_NOARG)
    impl::DecNum decnum;
    fData->quantity.toDecNum(decnum, status);
    decnum.toString(sink, status);
}

void FormattedNumber::getAllFieldPositionsImpl(FieldPositionIteratorHandler& fpih,
                                               UErrorCode& status) const {
    UPRV_FORMATTED_VALUE_METHOD_GUARD(UPRV_NOARG)
    fData->getAllFieldPositions(fpih, status);
}

MeasureUnit FormattedNumber::getOutputUnit(UErrorCode& status) const {
    UPRV_FORMATTED_VALUE_METHOD_GUARD(MeasureUnit())
    return fData->outputUnit;
}

UDisplayOptionsNounClass FormattedNumber::getNounClass(UErrorCode &status) const {
    UPRV_FORMATTED_VALUE_METHOD_GUARD(UDISPOPT_NOUN_CLASS_UNDEFINED);
    const char *nounClass = fData->gender;
    return udispopt_fromNounClassIdentifier(nounClass);
}

void FormattedNumber::getDecimalQuantity(impl::DecimalQuantity& output, UErrorCode& status) const {
    UPRV_FORMATTED_VALUE_METHOD_GUARD(UPRV_NOARG)
    output = fData->quantity;
}


impl::UFormattedNumberData::~UFormattedNumberData() = default;


UPRV_FORMATTED_VALUE_SUBCLASS_AUTO_IMPL(FormattedNumberRange)

#define UPRV_NOARG

void FormattedNumberRange::getDecimalNumbers(ByteSink& sink1, ByteSink& sink2, UErrorCode& status) const {
    UPRV_FORMATTED_VALUE_METHOD_GUARD(UPRV_NOARG)
    impl::DecNum decnum1;
    impl::DecNum decnum2;
    fData->quantity1.toDecNum(decnum1, status).toString(sink1, status);
    fData->quantity2.toDecNum(decnum2, status).toString(sink2, status);
}

UNumberRangeIdentityResult FormattedNumberRange::getIdentityResult(UErrorCode& status) const {
    UPRV_FORMATTED_VALUE_METHOD_GUARD(UNUM_IDENTITY_RESULT_NOT_EQUAL)
    return fData->identityResult;
}

const impl::UFormattedNumberRangeData* FormattedNumberRange::getData(UErrorCode& status) const {
    UPRV_FORMATTED_VALUE_METHOD_GUARD(nullptr)
    return fData;
}


impl::UFormattedNumberRangeData::~UFormattedNumberRangeData() = default;


} // namespace number
U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_FORMATTING */
