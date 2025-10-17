/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
 * Copyright (c) 1997-2016 International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef _INTLTESTDATETIMEPATTERNGENERATORAPI
#define _INTLTESTDATETIMEPATTERNGENERATORAPI

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/dtptngen.h"
#include "unicode/ustring.h"
#include "intltest.h"

/**
 * Test basic functionality of various API functions
 **/
class IntlTestDateTimePatternGeneratorAPI : public IntlTest {
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

private:
    /**
     * Performs tests on many API functions, see detailed comments in source code
     **/
    void testAPI(/* char* par */);
    void testOptions(/* char* par */);
    void testAllFieldPatterns(/* char* par */);
    void testStaticGetSkeleton(/* char* par */);
    void testC();
    void testSkeletonsWithDayPeriods();
    void testGetFieldDisplayNames();
    void testJjMapping();
    void test20640_HourCyclArsEnNH();
    void testFallbackWithDefaultRootLocale();
    void testGetDefaultHourCycle_OnEmptyInstance();
    void test_jConsistencyOddLocales();
    void testBestPattern();
    void testDateTimePatterns();
    void testISO8601();
    void testRegionOverride();
#if APPLE_ICU_CHANGES
// rdar://
   void testHorizontalInheritance();   // rdar://78420184
    void testPunjabiPattern(); // rdar://139506314
#endif  // APPLE_ICU_CHANGES

    // items for testDateTimePatterns();
    enum { kNumDateTimePatterns = 4 };
    typedef struct {
        const char* localeID;
        const UnicodeString expectPat[kNumDateTimePatterns];
    } DTPLocaleAndResults;
    void doDTPatternTest(DateTimePatternGenerator* dtpg, UnicodeString* skeletons, DTPLocaleAndResults* localeAndResultsPtr);
};

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
