/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
 * Copyright (c) 1997-2009, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef _INTLTESTDATEFORMATAPI
#define _INTLTESTDATEFORMATAPI

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"


/*
 * This is an API test, not a unit test.  It doesn't test very many cases, and doesn't
 * try to test the full functionality.  It just calls each function in the class and
 * verifies that it works on a basic level.
 */
class IntlTestDateFormatAPI: public IntlTest {
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

private:
    /**
     * Tests basic functionality of various generic API methods in DateFormat 
     */
    void testAPI(/* char* par */);
    /**
     * Test that the equals method works correctly.
     */
    void TestEquals();

    /**
     * Test that no parse or format methods are hidden.
     */
    void TestNameHiding();

    /**
     * Add better code coverage.
     */
    void TestCoverage();
};

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
