/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef __SOURCE_NUMBER_UTYPES_H__
#define __SOURCE_NUMBER_UTYPES_H__

#include "unicode/numberformatter.h"
#include "number_types.h"
#include "number_decimalquantity.h"
#include "formatted_string_builder.h"
#include "formattedval_impl.h"

U_NAMESPACE_BEGIN
namespace number::impl {

/** Helper function used in upluralrules.cpp */
const DecimalQuantity* validateUFormattedNumberToDecimalQuantity(
    const UFormattedNumber* uresult, UErrorCode& status);


/**
 * Struct for data used by FormattedNumber.
 *
 * This struct is held internally by the C++ version FormattedNumber since the member types are not
 * declared in the public header file.
 *
 * Exported as U_I18N_API for tests
 */
class U_I18N_API UFormattedNumberData : public FormattedValueStringBuilderImpl {
public:
    UFormattedNumberData() : FormattedValueStringBuilderImpl(kUndefinedField) {}
    virtual ~UFormattedNumberData();

    UFormattedNumberData(UFormattedNumberData&&) = default;
    UFormattedNumberData& operator=(UFormattedNumberData&&) = default;

    // The formatted quantity.
    DecimalQuantity quantity;

    // The output unit for the formatted quantity.
    // TODO(units,hugovdm): populate this correctly for the general case - it's
    // currently only implemented for the .usage() use case.
    MeasureUnit outputUnit;

    // The gender of the formatted output.
    const char *gender = "";
};

} // namespace number::impl
U_NAMESPACE_END

#endif //__SOURCE_NUMBER_UTYPES_H__
#endif /* #if !UCONFIG_NO_FORMATTING */
