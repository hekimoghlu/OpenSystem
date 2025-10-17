/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
 * Copyright (c) 1997-2013, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef _INTLTESTDATEFORMATSYMBOLS
#define _INTLTESTDATEFORMATSYMBOLS

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"

/**
 * Tests for DateFormatSymbols
 **/
class IntlTestDateFormatSymbols: public IntlTest {
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

private:
    /**
     * Test the API of DateFormatSymbols; primarily a simple get/set set.
     */
    void TestSymbols(/* char *par */);
    /**
     * Test getMonths.
     */
    void TestGetMonths();
    void TestGetMonths2();

    void TestGetWeekdays2();
    void TestGetEraNames();
    void TestGetSetSpecificItems();

    UBool UnicodeStringsArePrefixes(int32_t count, int32_t prefixLen, const UnicodeString *prefixArray, const UnicodeString *baseArray);
};

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
