/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
 * Copyright (c) 1997-2008, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/**
 * TestChoiceFormat is a third level test class
 */

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"


/**
 * tests Choice Format, functionality of examples, as well as API functionality
 **/
class TestChoiceFormat: public IntlTest {
    /** 
     *    tests basic functionality in a simple example
     **/
    void TestSimpleExample(); 
    /**
     *    tests functionality in a more complex example,
     *    and extensive API functionality.
     *    See verbose message output statements for specifically tested API
     **/
    void TestComplexExample();

    /**
     * Test new closure API
     */
    void TestClosures();

    /**
     * Test applyPattern
     */
    void TestPatterns();
    void TestChoiceFormatToPatternOverflow();

    void _testPattern(const char* pattern,
                      UBool isValid,
                      double v1, const char* str1,
                      double v2, const char* str2,
                      double v3, const char* str3);
    /** 
     *    runs tests in local functions:
     **/
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;
};

#endif /* #if !UCONFIG_NO_FORMATTING */
