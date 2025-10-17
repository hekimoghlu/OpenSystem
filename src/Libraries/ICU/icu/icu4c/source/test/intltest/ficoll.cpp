/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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
 * Copyright (c) 1997-2009,2014, International Business Machines
 * Corporation and others. All Rights Reserved.
 ********************************************************************/

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#ifndef _COLL
#include "unicode/coll.h"
#endif

#ifndef _TBLCOLL
#include "unicode/tblcoll.h"
#endif

#ifndef _UNISTR
#include "unicode/unistr.h"
#endif

#ifndef _SORTKEY
#include "unicode/sortkey.h"
#endif

#ifndef _FICOLL
#include "ficoll.h"
#endif

#include "sfwdchit.h"

CollationFinnishTest::CollationFinnishTest()
: myCollation(nullptr)
{
    UErrorCode status = U_ZERO_ERROR;
    myCollation = Collator::createInstance(Locale("fi", "FI", "", "collation=standard"),status);
}

CollationFinnishTest::~CollationFinnishTest()
{
    delete myCollation;
}

const char16_t CollationFinnishTest::testSourceCases[][CollationFinnishTest::MAX_TOKEN_LEN] = {
    {0x77, 0x61, 0x74, 0},  // "wat"
    {0x76, 0x61, 0x74, 0},  // "vat"
    {0x61, 0x00FC, 0x62, 0x65, 0x63, 0x6b, 0},
    {0x4c, 0x00E5, 0x76, 0x69, 0},
    {0x77, 0x61, 0x74, 0}   // "wat"
};

const char16_t CollationFinnishTest::testTargetCases[][CollationFinnishTest::MAX_TOKEN_LEN] = {
    {0x76, 0x61, 0x74, 0},  // "vat"
    {0x77, 0x61, 0x79, 0},
    {0x61, 0x78, 0x62, 0x65, 0x63, 0x6b, 0},
    {0x4c, 0x00E4, 0x77, 0x65, 0},
    {0x76, 0x61, 0x74, 0}   // "vat"
};

const Collator::EComparisonResult CollationFinnishTest::results[] = {
    Collator::GREATER,
    Collator::LESS,
    Collator::GREATER,
    Collator::LESS,
    // test primary > 4
    Collator::GREATER,  // v < w per cldrbug 6615
};

void CollationFinnishTest::TestTertiary(/* char* par */)
{
    int32_t i = 0;
    myCollation->setStrength(Collator::TERTIARY);
    for (i = 0; i < 4 ; i++) {
        doTest(myCollation, testSourceCases[i], testTargetCases[i], results[i]);
    }
}
void CollationFinnishTest::TestPrimary(/* char* par */)
{
    int32_t i;
    myCollation->setStrength(Collator::PRIMARY);
    for (i = 4; i < 5; i++) {
        doTest(myCollation, testSourceCases[i], testTargetCases[i], results[i]);
    }
}

void CollationFinnishTest::runIndexedTest( int32_t index, UBool exec, const char* &name, char* /*par*/ )
{
    if (exec) logln("TestSuite CollationFinnishTest: ");

    if((!myCollation) && exec) {
      dataerrln(__FILE__ " cannot test - failed to create collator.");
      name = "some test";
      return;
    }
    switch (index) {
        case 0: name = "TestPrimary";   if (exec)   TestPrimary(/* par */); break;
        case 1: name = "TestTertiary";  if (exec)   TestTertiary(/* par */); break;
        default: name = ""; break;
    }
}

#endif /* #if !UCONFIG_NO_COLLATION */
