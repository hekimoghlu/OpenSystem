/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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

#ifndef _PLURALFORMATTEST
#define _PLURALFORMATTEST

#include "unicode/utypes.h"
#include "unicode/plurrule.h"
#include "unicode/plurfmt.h"


#if !UCONFIG_NO_FORMATTING

#include "intltest.h"

/**
 * Test basic functionality of various API functions
 **/
class PluralFormatTest : public IntlTest {
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

private:
    /**
     * Performs tests on many API functions, see detailed comments in source code
     **/
    void pluralFormatBasicTest(/* char* par */);
    void pluralFormatUnitTest(/* char* par */);
    void pluralFormatLocaleTest(/* char* par */);
    void pluralFormatExtendedTest();
    void pluralFormatExtendedParseTest();
    void ordinalFormatTest();
    void TestDecimals();
    void numberFormatTest(PluralFormat* plFmt, 
                          NumberFormat *numFmt, 
                          int32_t start, 
                          int32_t end, 
                          UnicodeString* numOddAppendStr,
                          UnicodeString* numEvenAppendStr, 
                          UBool overwrite, // overwrite the numberFormat.format result
                          UnicodeString *message);
    void helperTestResults(const char** localeArray, 
                           int32_t capacityOfArray, 
                           UnicodeString& testPattern, 
                           int8_t *expectingResults);
};

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
