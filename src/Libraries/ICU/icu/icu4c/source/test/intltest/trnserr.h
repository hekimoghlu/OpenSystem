/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
 * Copyright (c) 2001-2005, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/************************************************************************
*   This test program is intended for testing error conditions of the 
*   transliterator APIs to make sure the exceptions are raised where
*   necessary.
*
*   Date        Name        Description
*   11/14/2001  hshih       Creation.
* 
************************************************************************/


#ifndef TRNSERR_H
#define TRNSERR_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/translit.h"
#include "intltest.h"

/**
 * @test
 * @summary Error condition tests of Transliterator
 */
class TransliteratorErrorTest : public IntlTest {
public:
    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par=nullptr) override;

    /*Tests the returned error codes on all the APIs according to the API documentation. */
    void TestTransliteratorErrors();
    
    void TestUnicodeSetErrors();

    //void TestUniToHexErrors();

    void TestRBTErrors();

    //void TestHexToUniErrors();

    // JitterBug 4452, for coverage.  The reason to put this method here is 
    //  this class is comparable smaller than other Transliterator*Test classes
    void TestCoverage();

};

#endif /* #if !UCONFIG_NO_TRANSLITERATION */

#endif
