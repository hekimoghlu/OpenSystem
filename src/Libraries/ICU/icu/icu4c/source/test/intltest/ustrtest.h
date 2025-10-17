/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
 * Copyright (c) 1997-2015, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef UNICODESTRINGTEST_H
#define UNICODESTRINGTEST_H

#include "unicode/locid.h"
#include "unicode/unistr.h"
#include "intltest.h"

U_NAMESPACE_BEGIN

class Appendable;

U_NAMESPACE_END

/**
 * Perform API and functionality tests for class UnicodeString
 **/
class UnicodeStringTest: public IntlTest {
public:
    UnicodeStringTest() {}
    virtual ~UnicodeStringTest();
    
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    /**
     * Test some basic methods (insert, remove, replace, ...)
     **/
    void TestBasicManipulation();
    /**
     * Test the methods for comparison
     **/
    void TestCompare();
    /**
     * Test the methods for extracting
     **/
    void TestExtract();
    /**
     * More extensively test methods for removing and replacing
     **/
    void TestRemoveReplace();
    /**
     * Test language specific case conversions
     **/
    void TestSearching();
    /**
     * Test methods for padding, trimmimg and truncating
     **/
    void TestSpacePadding();
    /**
     * Test methods startsWith and endsWith
     **/
    void TestPrefixAndSuffix();
    void TestStartsWithAndEndsWithNulTerminated();
    /**
     * Test method findAndReplace
     **/
    void TestFindAndReplace();
    /**
     * Test method reverse
     **/
    void TestReverse();
    /**
     * Test a few miscellaneous methods (isBogus, hashCode,...)
     **/
    void TestMiscellaneous();
    /**
     * Test the functionality of allocating UnicodeStrings on the stack
     **/
    void TestStackAllocation();
    /**
     * Test the unescape() function.
     */
    void TestUnescape();

    void _testUnicodeStringHasMoreChar32Than(const UnicodeString &s, int32_t start, int32_t length, int32_t number);
    void TestCountChar32();
    void TestBogus();
    void TestStringEnumeration();
    void TestNameSpace();
    void TestUTF32();
    void TestUTF8();
    void TestReadOnlyAlias();
    void doTestAppendable(UnicodeString &dest, Appendable &app);
    void TestAppendable();
    void TestUnicodeStringImplementsAppendable();
    void TestSizeofUnicodeString();
    void TestMoveSwap();

    void TestUInt16Pointers();
    void TestWCharPointers();
    void TestNullPointers();
    void TestUnicodeStringInsertAppendToSelf();
    void TestLargeAppend();
    void TestU16StringView();
    void TestWStringView();
};

#endif
