/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
/********************************************************************
 * COPYRIGHT:
 * Copyright (c) 1997-2013, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef _PluralRulesTest
#define _PluralRulesTest

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"
#include "number_decimalquantity.h"
#include "unicode/localpointer.h"
#include "unicode/plurrule.h"

/**
 * Test basic functionality of various API functions
 **/
class PluralRulesTest : public IntlTest {
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

private:
    /**
     * Performs tests on many API functions, see detailed comments in source code
     **/
    void testAPI();
    void testGetUniqueKeywordValue();
    void testGetSamples();
    void testGetDecimalQuantitySamples();
    void testGetOrAddSamplesFromString();
    void testGetOrAddSamplesFromStringCompactNotation();
    void testSamplesWithExponent();
    void testSamplesWithCompactNotation();
    void testWithin();
    void testGetAllKeywordValues();
    void testCompactDecimalPluralKeyword();
    void testDoubleValue();
    void testLongValue();
    void testScientificPluralKeyword();
    void testOrdinal();
    void testSelect();
    void testSelectRange();
    void testAvailableLocales();
    void testParseErrors();
    void testFixedDecimal();
    void testSelectTrailingZeros();
    void testLocaleExtension();
    void testDoubleEqualSign();
    void test22638LongNumberValue();

    void assertRuleValue(const UnicodeString& rule, double expected);
    void assertRuleKeyValue(const UnicodeString& rule, const UnicodeString& key,
                            double expected);
    void checkNewSamples(UnicodeString description, 
                         const LocalPointer<PluralRules> &test,
                         UnicodeString keyword,
                         UnicodeString samplesString,
                         ::icu::number::impl::DecimalQuantity firstInRange);
    UnicodeString getPluralKeyword(const LocalPointer<PluralRules> &rules,
                                   Locale locale, double number, const char16_t* skeleton);
    void checkSelect(const LocalPointer<PluralRules> &rules, UErrorCode &status, 
                                  int32_t line, const char *keyword, ...);
    void compareLocaleResults(const char* loc1, const char* loc2, const char* loc3);
};

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
