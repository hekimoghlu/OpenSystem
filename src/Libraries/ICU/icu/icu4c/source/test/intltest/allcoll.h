/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
/***********************************************************************
 * COPYRIGHT: 
 * Copyright (c) 1997-2004, International Business Machines Corporation
 * and others. All Rights Reserved.
 ***********************************************************************/

/**
 * CollationDummyTest is a third level test class.  This tests creation of 
 * a customized collator object.  For example, number 1 to be sorted 
 * equlivalent to word 'one'.
 */

#ifndef _ALLCOLL
#define _ALLCOLL

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "unicode/tblcoll.h"
#include "tscoll.h"

class CollationDummyTest: public IntlTestCollator {
public:
    // If this is too small for the test data, just increase it.
    // Just don't make it too large, otherwise the executable will get too big
    enum EToken_Len { MAX_TOKEN_LEN = 16 };

    CollationDummyTest();
    virtual ~CollationDummyTest();
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* /*par = nullptr */) override;

    // perform test with strength PRIMARY
    void TestPrimary(/* char* par */);

    // perform test with strength SECONDARY
    void TestSecondary(/* char* par */);

    // perform test with strength tertiary
    void TestTertiary(/* char* par */);

    // perform extra tests
    void TestExtra(/* char* par */);

    void TestIdentical();

    void TestJB581();

private:
    static const  Collator::EComparisonResult results[];

    RuleBasedCollator *myCollation;
};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
