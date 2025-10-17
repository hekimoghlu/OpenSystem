/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
 * CollationFinnishTest is a third level test class. This tests the locale
 * specific primary and tertiary rules. For example, a-ring sorts after z
 * and w and v are equivalent.
 */

#ifndef _FICOLL
#define _FICOLL

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "tscoll.h"

class CollationFinnishTest: public IntlTestCollator {
public:
    // If this is too small for the test data, just increase it.
    // Just don't make it too large, otherwise the executable will get too big
    enum EToken_Len { MAX_TOKEN_LEN = 16 };

    CollationFinnishTest();
    virtual ~CollationFinnishTest();
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    // perform tests with strength PRIMARY
    void TestPrimary(/* char* par */);

    // perform test with strength TERTIARY
    void TestTertiary(/* char* par */);

private:
    static const char16_t testSourceCases[][MAX_TOKEN_LEN];
    static const char16_t testTargetCases[][MAX_TOKEN_LEN];
    static const Collator::EComparisonResult results[];

    Collator *myCollation;
};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
