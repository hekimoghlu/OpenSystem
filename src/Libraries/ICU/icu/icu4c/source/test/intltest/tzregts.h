/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
 * Copyright (c) 1998-2005, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef _TIMEZONEREGRESSIONTEST_
#define _TIMEZONEREGRESSIONTEST_
 
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/timezone.h"
#include "unicode/gregocal.h"
#include "unicode/simpletz.h"
#include "intltest.h"



/** 
 * Performs regression test for Calendar
 **/
class TimeZoneRegressionTest: public IntlTest {    
    
    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
public:
    
    void Test4052967();
    void Test4073209();
    void Test4073215();
    void Test4084933();
    void Test4096952();
    void Test4109314();
    void Test4126678();
    void Test4151406();
    void Test4151429();
    void Test4154537();
    void Test4154542();
    void Test4154650();
    void Test4154525();
    void Test4162593();
    void Test4176686();
    void TestJ186();
    void TestJ449();
    void TestJDK12API();
    void Test4184229();
    UBool checkCalendar314(GregorianCalendar *testCal, TimeZone *testTZ);
    void TestNegativeDaylightSaving();


protected:
    UDate findTransitionBinary(const SimpleTimeZone& tz, UDate min, UDate max);
    UDate findTransitionStepwise(const SimpleTimeZone& tz, UDate min, UDate max);
    UBool failure(UErrorCode status, const char* msg);
};

#endif /* #if !UCONFIG_NO_FORMATTING */
 
#endif // _CALENDARREGRESSIONTEST_
//eof
