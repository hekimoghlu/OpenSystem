/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
 * Copyright (c) 1997-2003, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/**
 * CollationMonkeyTest is a third level test class.  This tests the random 
 * substrings of the default test strings to verify if the compare and 
 * sort key algorithm works correctly.  For example, any string is always
 * less than the string itself appended with any character.
 */

#ifndef _MNKYTST
#define _MNKYTST

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "tscoll.h"

class CollationMonkeyTest: public IntlTestCollator {
public:
    // If this is too small for the test data, just increase it.
    // Just don't make it too large, otherwise the executable will get too big
    enum EToken_Len { MAX_TOKEN_LEN = 16 };

    CollationMonkeyTest();
    virtual ~CollationMonkeyTest();
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    // utility function used in tests, returns absolute value
    int32_t checkValue(int32_t value);

    // perform monkey tests using Collator::compare
    void TestCompare(/* char* par */);

    // perform monkey tests using CollationKey::compareTo
    void TestCollationKey(/* char* par */);

    void TestRules(/* char* par */);

private:
    void report(UnicodeString& s, UnicodeString& t, int32_t result, int32_t revResult);

    const UnicodeString source;

    Collator *myCollator;
};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
