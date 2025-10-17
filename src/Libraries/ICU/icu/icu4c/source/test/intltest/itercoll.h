/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
 * Copyright (c) 1997-2001, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/**
 * Collation Iterator tests.
 * (Let me reiterate my position...)
 */

#ifndef _ITERCOLL
#define _ITERCOLL

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "unicode/tblcoll.h"
#include "unicode/coleitr.h"
#include "tscoll.h"

class CollationIteratorTest: public IntlTestCollator
{
public:

    // If this is too small for the test data, just increase it.
    // Just don't make it too large, otherwise the executable will get too big
    enum EToken_Len { MAX_TOKEN_LEN = 16 };

    CollationIteratorTest();
    virtual ~CollationIteratorTest();

    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par = nullptr) override;

    /**
    * Test that results from CollationElementIterator.next is equivalent to
    * the reversed results from CollationElementIterator.previous, for the set 
    * of BMP characters.
    */
    void TestUnicodeChar();

    /**
     * Test for CollationElementIterator.previous()
     *
     * @bug 4108758 - Make sure it works with contracting characters
     * 
     */
    void TestPrevious(/* char* par */);
    
    /**
     * Test for getOffset() and setOffset()
     */
    void TestOffset(/* char* par */);

    /**
     * Test for setText()
     */
    void TestSetText(/* char* par */);
    
    /** @bug 4108762
     * Test for getMaxExpansion()
     */
    void TestMaxExpansion(/* char* par */);

    /*
     * @bug 4157299
     */
    void TestClearBuffers(/* char* par */);

    /**
     * Testing the assignment operator
     */
    void TestAssignment();

    /**
     * Testing the constructors
     */
    void TestConstructors();

    /**
    * Testing the strength order functionality
    */
    void TestStrengthOrder();
    
    //------------------------------------------------------------------------
    // Internal utilities
    //

private:

    struct ExpansionRecord
    {
        char16_t character;
        int32_t count;
    };

    /**
     * Verify that getMaxExpansion works on a given set of collation rules
     */
    void verifyExpansion(UnicodeString rules, ExpansionRecord tests[], int32_t testCount);
    
    /**
     * Return a string containing all of the collation orders
     * returned by calls to next on the specified iterator
     */
    UnicodeString &orderString(CollationElementIterator &iter, UnicodeString &target);

    void assertEqual(CollationElementIterator &i1, CollationElementIterator &i2);

    RuleBasedCollator *en_us;
    const UnicodeString test1;
    const UnicodeString test2;

};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
