/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
*   Date        Name        Description
*   1/03/2000   Madhu        Creation.
************************************************************************/

#ifndef TRANSAPI_H
#define TRANSAPI_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/translit.h"
#include "intltest.h"

/**
 * @test
 * @summary General test of Transliterator
 */
class TransliteratorAPITest : public IntlTest {
public:
    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par=nullptr) override;

    /*Tests the function getDisplayName() */
    void TestGetDisplayName();

    void TestgetID();

    void TestgetInverse();

    void TestClone();

    void TestTransliterate1();

    void TestTransliterate2();

    void TestTransliterate3();

    void TestSimpleKeyboardTransliterator();

    void TestKeyboardTransliterator1();

    void TestKeyboardTransliterator2();

    void TestKeyboardTransliterator3();

    void TestGetAdoptFilter();

    void TestNullTransliterator();

    void TestRegisterUnregister();

    void TestLatinDevanagari();
    
    void TestDevanagariLatinRT();

    void TestUnicodeFunctor();

    /*Internal functions used*/
    void doTest(const UnicodeString& , const UnicodeString& , const UnicodeString& );

    void keyboardAux(Transliterator*, UnicodeString[] , UnicodeString&, int32_t, int32_t);

    void displayOutput(const UnicodeString&, const UnicodeString&, UnicodeString&,
                       UTransPosition&);

    void callEverything(const Transliterator *t, int line);

};

#endif /* #if !UCONFIG_NO_TRANSLITERATION */

#endif
