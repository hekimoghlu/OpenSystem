/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
/***********************************************************************
************************************************************************
*   Date        Name        Description
*   03/09/2000   Madhu        Creation.
************************************************************************/

#ifndef CPDTRTST_H
#define CPDTRTST_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/translit.h"
#include "cpdtrans.h"
#include "intltest.h"

/**
 * @test
 * @summary General test of Compound Transliterator
 */
class CompoundTransliteratorTest : public IntlTest {
public:
    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par=nullptr) override;

    /*Tests the constructors */
    void TestConstruction();
    /*Tests the function clone, and operator==()*/
    void TestCloneEqual();
    /*Tests the function getCount()*/
    void TestGetCount();
    /*Tests the function getTransliterator() and setTransliterators() and adoptTransliterators()*/
    void TestGetSetAdoptTransliterator();
    /*Tests the function handleTransliterate()*/
    void TestTransliterate();

    //======================================================================
    // Support methods
    //======================================================================

    /**
     * Splits a UnicodeString
     */
    UnicodeString* split(const UnicodeString& str, char16_t seperator, int32_t& count);

    void expect(const CompoundTransliterator& t,
                const UnicodeString& source,
                const UnicodeString& expectedResult);

    void expectAux(const UnicodeString& tag,
                   const UnicodeString& summary, UBool pass,
                   const UnicodeString& expectedResult);


};

#endif /* #if !UCONFIG_NO_TRANSLITERATION */

#endif
