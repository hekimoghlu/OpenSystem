/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
*******************************************************************************
* Copyright (C) 2007-2015, International Business Machines Corporation and    *
* others. All Rights Reserved.                                                *
*******************************************************************************
*/

#ifndef _TIMEZONEFORMATTEST_
#define _TIMEZONEFORMATTEST_

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"

class TimeZoneFormatTest : public IntlTest {
  public:
    // IntlTest override
    void runIndexedTest(int32_t index, UBool exec, const char*& name, char* par) override;

    void TestTimeZoneRoundTrip();
    void TestTimeRoundTrip();
    void TestParse();
    void TestISOFormat();
    void TestFormat();
    void TestFormatTZDBNames();
    void TestFormatCustomZone();
    void TestFormatTZDBNamesAllZoneCoverage();
    void TestAdoptDefaultThreadSafe();
    void TestCentralTime();
    void TestBogusLocale();
    void Test22614GetMetaZoneNamesNotCrash();
    void Test22615NonASCIIID();

    void RunTimeRoundTripTests(int32_t threadNumber);
    void RunAdoptDefaultThreadSafeTests(int32_t threadNumber);
};

#endif /* #if !UCONFIG_NO_FORMATTING */
 
#endif // _TIMEZONEFORMATTEST_
