/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
* Copyright (c) 2004-2014 International Business Machines
* Corporation and others.  All Rights Reserved.
**********************************************************************
* Author: Alan Liu
* Created: April 20, 2004
* Since: ICU 3.0
**********************************************************************
*/
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "currfmt.h"
#include "unicode/numfmt.h"
#include "unicode/curramt.h"

U_NAMESPACE_BEGIN

CurrencyFormat::CurrencyFormat(const Locale& locale, UErrorCode& ec) :
    MeasureFormat(locale, UMEASFMT_WIDTH_WIDE, ec)
{
}

CurrencyFormat::CurrencyFormat(const CurrencyFormat& other) :
    MeasureFormat(other)
{
}

CurrencyFormat::~CurrencyFormat() {
}

CurrencyFormat* CurrencyFormat::clone() const {
    return new CurrencyFormat(*this);
}

UnicodeString& CurrencyFormat::format(const Formattable& obj,
                                      UnicodeString& appendTo,
                                      FieldPosition& pos,
                                      UErrorCode& ec) const
{
    return getCurrencyFormatInternal().format(obj, appendTo, pos, ec);
}

void CurrencyFormat::parseObject(const UnicodeString& source,
                                 Formattable& result,
                                 ParsePosition& pos) const
{
    CurrencyAmount* currAmt = getCurrencyFormatInternal().parseCurrency(source, pos);
    if (currAmt != nullptr) {
        result.adoptObject(currAmt);
    }
}

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(CurrencyFormat)

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_FORMATTING */
