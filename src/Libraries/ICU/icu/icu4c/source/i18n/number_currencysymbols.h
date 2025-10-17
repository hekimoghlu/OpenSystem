/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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
#ifndef __SOURCE_NUMBER_CURRENCYSYMBOLS_H__
#define __SOURCE_NUMBER_CURRENCYSYMBOLS_H__

#include "numparse_types.h"
#include "charstr.h"
#include "number_decimfmtprops.h"

U_NAMESPACE_BEGIN
namespace number::impl {

// Exported as U_I18N_API for tests
class U_I18N_API CurrencySymbols : public UMemory {
  public:
    CurrencySymbols() = default; // default constructor: leaves class in valid but undefined state

    /** Creates an instance in which all symbols are loaded from data. */
    CurrencySymbols(CurrencyUnit currency, const Locale& locale, UErrorCode& status);

    /** Creates an instance in which some symbols might be pre-populated. */
    CurrencySymbols(CurrencyUnit currency, const Locale& locale, const DecimalFormatSymbols& symbols,
                    UErrorCode& status);

    const char16_t* getIsoCode() const;

    UnicodeString getNarrowCurrencySymbol(UErrorCode& status) const;

    UnicodeString getFormalCurrencySymbol(UErrorCode& status) const;

    UnicodeString getVariantCurrencySymbol(UErrorCode& status) const;

    UnicodeString getCurrencySymbol(UErrorCode& status) const;

    UnicodeString getIntlCurrencySymbol(UErrorCode& status) const;

    UnicodeString getPluralName(StandardPlural::Form plural, UErrorCode& status) const;

    bool hasEmptyCurrencySymbol() const;

  protected:
    // Required fields:
    CurrencyUnit fCurrency;
    CharString fLocaleName;

    // Optional fields:
    UnicodeString fCurrencySymbol;
    UnicodeString fIntlCurrencySymbol;

    UnicodeString loadSymbol(UCurrNameStyle selector, UErrorCode& status) const;
};


/**
 * Resolves the effective currency from the property bag.
 */
CurrencyUnit
resolveCurrency(const DecimalFormatProperties& properties, const Locale& locale, UErrorCode& status);

} // namespace number::impl
U_NAMESPACE_END

#endif //__SOURCE_NUMBER_CURRENCYSYMBOLS_H__
#endif /* #if !UCONFIG_NO_FORMATTING */
