/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
**********************************************************************
* Copyright (c) 2004, International Business Machines
* Corporation and others.  All Rights Reserved.
**********************************************************************
* Author: Alan Liu
* Created: April 26, 2004
* Since: ICU 3.0
**********************************************************************
*/
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/curramt.h"
#include "unicode/currunit.h"

U_NAMESPACE_BEGIN

CurrencyAmount::CurrencyAmount(const Formattable& amount, ConstChar16Ptr isoCode,
                               UErrorCode& ec) :
    Measure(amount, new CurrencyUnit(isoCode, ec), ec) {
}

CurrencyAmount::CurrencyAmount(double amount, ConstChar16Ptr isoCode,
                               UErrorCode& ec) :
    Measure(Formattable(amount), new CurrencyUnit(isoCode, ec), ec) {
}

CurrencyAmount::CurrencyAmount(const CurrencyAmount& other) :
    Measure(other) {
}

CurrencyAmount& CurrencyAmount::operator=(const CurrencyAmount& other) {
    Measure::operator=(other);
    return *this;
}

CurrencyAmount* CurrencyAmount::clone() const {
    return new CurrencyAmount(*this);
}

CurrencyAmount::~CurrencyAmount() {
}

const CurrencyUnit& CurrencyAmount::getCurrency() const {
    return static_cast<const CurrencyUnit&>(getUnit());
}

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(CurrencyAmount)

U_NAMESPACE_END

#endif // !UCONFIG_NO_FORMATTING
