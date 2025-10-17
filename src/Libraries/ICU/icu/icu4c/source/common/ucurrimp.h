/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
* Copyright (c) 2002-2016, International Business Machines
* Corporation and others.  All Rights Reserved.
**********************************************************************
*/

#ifndef _UCURR_IMP_H_
#define _UCURR_IMP_H_

#include "unicode/utypes.h"
#include "unicode/unistr.h"
#include "unicode/parsepos.h"
#include "unicode/uniset.h"

/**
 * Internal method.  Given a currency ISO code and a locale, return
 * the "static" currency name.  This is usually the same as the
 * UCURR_SYMBOL_NAME, but if the latter is a choice format, then the
 * format is applied to the number 2.0 (to yield the more common
 * plural) to return a static name.
 *
 * This is used for backward compatibility with old currency logic in
 * DecimalFormat and DecimalFormatSymbols.
 */
U_CAPI void
uprv_getStaticCurrencyName(const UChar* iso, const char* loc,
                           icu::UnicodeString& result, UErrorCode& ec);

/**
 * Attempt to parse the given string as a currency, either as a
 * display name in the given locale, or as a 3-letter ISO 4217
 * code.  If multiple display names match, then the longest one is
 * selected.  If both a display name and a 3-letter ISO code
 * match, then the display name is preferred, unless it's length
 * is less than 3.
 *
 * The parameters must not be NULL.
 *
 * @param locale the locale of the display names to match
 * @param text the text to parse
 * @param pos input-output position; on input, the position within
 * text to match; must have 0 <= pos.getIndex() < text.length();
 * on output, the position after the last matched character. If
 * the parse fails, the position in unchanged upon output.
 * @param type currency type to parse against, LONG_NAME only or not
 * @param partialMatchLen The length of the longest matching prefix;
 * this may be nonzero even if no full currency was matched.
 * @return the ISO 4217 code, as a string, of the best match, or
 * null if there is no match
 *
 * @internal
 */
U_CAPI void
uprv_parseCurrency(const char* locale,
                   const icu::UnicodeString& text,
                   icu::ParsePosition& pos,
                   int8_t type,
                   int32_t* partialMatchLen,
                   UChar* result,
                   UErrorCode& ec);

/**
 * Puts all possible first-characters of a currency into the
 * specified UnicodeSet.
 *
 * @param locale the locale of the display names of interest
 * @param result the UnicodeSet to which to add the starting characters
 */
void uprv_currencyLeads(const char* locale, icu::UnicodeSet& result, UErrorCode& ec);



#endif /* #ifndef _UCURR_IMP_H_ */

//eof
