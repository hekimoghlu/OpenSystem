/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
/****************************************************************************
 * COPYRIGHT: 
 * Copyright (c) 2001-2008, International Business Machines Corporation and others
 * All Rights Reserved.
 ***************************************************************************/

#ifndef _STRSRCH_H
#define _STRSRCH_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION 

#include "unicode/tblcoll.h"
#include "unicode/brkiter.h"
#include "intltest.h"
#include "unicode/usearch.h"

struct SearchData;
typedef struct SearchData SearchData;

class StringSearchTest: public IntlTest 
{
public:
    StringSearchTest();
    virtual ~StringSearchTest();

    void runIndexedTest(int32_t index, UBool exec, const char* &name, 
                        char* par = nullptr) override;
#if !UCONFIG_NO_BREAK_ITERATION
private:
    RuleBasedCollator *m_en_us_; 
    RuleBasedCollator *m_fr_fr_;
    RuleBasedCollator *m_de_;
    RuleBasedCollator *m_es_;
    BreakIterator     *m_en_wordbreaker_;
    BreakIterator     *m_en_characterbreaker_;

    RuleBasedCollator * getCollator(const char *collator);
    BreakIterator     * getBreakIterator(const char *breaker);
    char              * toCharString(const UnicodeString &text);
    Collator::ECollationStrength getECollationStrength(
                                   const UCollationStrength &strength) const;
    UBool           assertEqualWithStringSearch(      StringSearch *strsrch,
                                                const SearchData   *search);
    UBool           assertEqual(const SearchData *search);
    UBool           assertCanonicalEqual(const SearchData *search);
    UBool           assertEqualWithAttribute(const SearchData *search, 
                                            USearchAttributeValue canonical,
                                            USearchAttributeValue overlap);
    void TestOpenClose();
    void TestInitialization();
    void TestBasic();
    void TestNormExact();
    void TestStrength();
#if !UCONFIG_NO_BREAK_ITERATION
    void TestBreakIterator();
#endif
    void TestVariable();
    void TestOverlap();
    void TestCollator();
    void TestPattern();
    void TestText();
    void TestCompositeBoundaries();
    void TestGetSetOffset();
    void TestGetSetAttribute();
    void TestGetMatch();
    void TestSetMatch();
    void TestReset();
    void TestSupplementary();
    void TestContraction();
    void TestIgnorable();
    void TestCanonical();
    void TestNormCanonical();
    void TestStrengthCanonical();
#if !UCONFIG_NO_BREAK_ITERATION
    void TestBreakIteratorCanonical();
#endif
    void TestVariableCanonical();
    void TestOverlapCanonical();
    void TestCollatorCanonical();
    void TestPatternCanonical();
    void TestTextCanonical();
    void TestCompositeBoundariesCanonical();
    void TestGetSetOffsetCanonical();
    void TestSupplementaryCanonical();
    void TestContractionCanonical();
    void TestUClassID();
    void TestSubclass();
    void TestCoverage();
    void TestDiacriticMatch();
#endif
};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
