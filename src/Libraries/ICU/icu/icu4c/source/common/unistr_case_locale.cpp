/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
*******************************************************************************
*   Copyright (C) 2011, International Business Machines
*   Corporation and others.  All Rights Reserved.
*******************************************************************************
*   file name:  unistr_case_locale.cpp
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2011may31
*   created by: Markus W. Scherer
*
*   Locale-sensitive case mapping functions (ones that call uloc_getDefault())
*   were moved here to break dependency cycles among parts of the common library.
*/

#include "unicode/utypes.h"
#include "unicode/locid.h"
#include "unicode/ucasemap.h"
#include "unicode/unistr.h"
#include "ucasemap_imp.h"

U_NAMESPACE_BEGIN

//========================================
// Write implementation
//========================================

UnicodeString &
UnicodeString::toLower() {
  return caseMap(ustrcase_getCaseLocale(nullptr), 0,
                 UCASEMAP_BREAK_ITERATOR_NULL ustrcase_internalToLower);
}

UnicodeString &
UnicodeString::toLower(const Locale &locale) {
  return caseMap(ustrcase_getCaseLocale(locale.getBaseName()), 0,
                 UCASEMAP_BREAK_ITERATOR_NULL ustrcase_internalToLower);
}

UnicodeString &
UnicodeString::toUpper() {
  return caseMap(ustrcase_getCaseLocale(nullptr), 0,
                 UCASEMAP_BREAK_ITERATOR_NULL ustrcase_internalToUpper);
}

UnicodeString &
UnicodeString::toUpper(const Locale &locale) {
  return caseMap(ustrcase_getCaseLocale(locale.getBaseName()), 0,
                 UCASEMAP_BREAK_ITERATOR_NULL ustrcase_internalToUpper);
}

U_NAMESPACE_END
