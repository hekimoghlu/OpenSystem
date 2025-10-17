/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
 * Copyright (c) 1999-2016 International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/************************************************************************
*   Date        Name        Description
*   12/14/99    Madhu        Creation.
************************************************************************/



#ifndef RBBIAPITEST_H
#define RBBIAPITEST_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_BREAK_ITERATION

#include "intltest.h"
#include "unicode/rbbi.h"

/**
 * API Test the RuleBasedBreakIterator class
 */
class RBBIAPITest: public IntlTest {
public:
   
    
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;
    /**
     * Tests Constructor behaviour of RuleBasedBreakIterator
     **/
   // void TestConstruction();   
    /**
     * Tests clone() and equals() methods of RuleBasedBreakIterator         
     **/
    void TestCloneEquals();
    /**
     * Tests toString() method of RuleBasedBreakIterator
     **/
    void TestgetRules();
    /**
     * Tests the method hashCode() of RuleBasedBreakIterator
     **/
    void TestHashCode();
     /**
      * Tests the methods getText() and setText() of RuleBasedBreakIterator
      **/
    void TestGetSetAdoptText();
     /**
      * Testing the iteration methods of RuleBasedBreakIterator
      **/
    void TestIteration();

    void TestFilteredBreakIteratorBuilder();

    /**
     * Tests creating RuleBasedBreakIterator from rules strings.
     **/
    void TestBuilder();

    void TestRoundtripRules();

    void RoundtripRule(const char *dataFile);

    /**
     * Test getting and using binary (compiled) rules.
     **/
    void TestGetBinaryRules();

    /**
     * Tests grouping effect of 'single quotes' in rules.
     **/
    void TestQuoteGrouping();

    /**
     *  Tests word break status returns.
     */
    void TestRuleStatus();
    void TestRuleStatusVec();

    void TestBug2190();
    void TestBug22580();

    void TestBoilerPlate();

    void TestRegistration();

    void TestRefreshInputText();

    /**
     *Internal subroutines
     **/
    /* Internal subroutine used by TestIsBoundary() */ 
    void doBoundaryTest(BreakIterator& bi, UnicodeString& text, int32_t *boundaries);

    /*Internal subroutine used for comparison of expected and acquired results */
    void doTest(UnicodeString& testString, int32_t start, int32_t gotoffset, int32_t expectedOffset, const char* expected);


};

#endif /* #if !UCONFIG_NO_BREAK_ITERATION */

#endif
