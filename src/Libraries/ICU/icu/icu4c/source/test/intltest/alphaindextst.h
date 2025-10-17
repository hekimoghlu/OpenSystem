/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
 * Copyright (c) 2012-2014, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
//
//   file:  alphaindextst.h
//          Alphabetic Index Tests.
//

#ifndef ALPHAINDEXTST_H
#define ALPHAINDEXTST_H

#include "unicode/uscript.h"
#include "intltest.h"

class AlphabeticIndexTest: public IntlTest {
public:
    AlphabeticIndexTest();
    virtual ~AlphabeticIndexTest();

    virtual void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    virtual void APITest();
    virtual void ManyLocalesTest();
    virtual void HackPinyinTest();
    virtual void TestBug9009();
    void TestIndexCharactersList();
    /**
     * Test AlphabeticIndex vs. root with script reordering.
     */
    void TestHaniFirst();
    /**
     * Test AlphabeticIndex vs. Pinyin with script reordering.
     */
    void TestPinyinFirst();
    /**
     * Test labels with multiple primary weights.
     */
    void TestSchSt();
    /**
     * With no real labels, there should be only the underflow label.
     */
    void TestNoLabels();
    /**
     * Test with the Bopomofo-phonetic tailoring.
     */
    void TestChineseZhuyin();
    void TestJapaneseKanji();
    void TestChineseUnihan();

    void testHasBuckets();
    void checkHasBuckets(const Locale &locale, UScriptCode script);
};

#endif
