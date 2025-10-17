/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
/*
**********************************************************************
*   Copyright (C) 1999-2003, International Business Machines
*   Corporation and others.  All Rights Reserved.
**********************************************************************
*   Date        Name        Description
*   12/09/99    aliu        Ported from Java.
**********************************************************************
*/

#ifndef COLLATIONTHAITEST_H
#define COLLATIONTHAITEST_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "tscoll.h"

class CollationThaiTest : public IntlTestCollator {
    Collator* coll; // Thai collator

public:

    CollationThaiTest();
    virtual ~CollationThaiTest();

    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;
    
private:

    /**
     * Read the external dictionary file, which is already in proper
     * sorted order, and confirm that the collator compares each line as
     * preceding the following line.
     */
    void TestDictionary();
    
    /**
     * Odd corner conditions taken from "How to Sort Thai Without Rewriting Sort",
     * by Doug Cooper, http://seasrc.th.net/paper/thaisort.zip
     */
    void TestCornerCases();
    
    /**
     * Read the external names list, and confirms that the collator 
     * gets the same results when comparing lines one to another
     * using regular and iterative comparison.
     */
    void TestNamesList();

    /** 
     * test that invalid Thai sorts properly
     */
    void TestInvalidThai();

    /** 
     * test that reording is done properly
     */
    void TestReordering();

private:

    void compareArray(Collator& c, const char* tests[],
                      int32_t testsLength);

    int8_t sign(int32_t i);
    
    /**
     * Set a UnicodeString corresponding to the given string.  Use
     * UnicodeString and the default converter, unless we see the sequence
     * "\\u", in which case we interpret the subsequent escape.
     */
    UnicodeString& parseChars(UnicodeString& result,
                              const char* chars);
};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
