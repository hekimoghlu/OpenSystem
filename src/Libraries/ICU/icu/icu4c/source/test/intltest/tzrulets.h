/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
* Copyright (C) 2007-2011, International Business Machines Corporation and    *
* others. All Rights Reserved.                                                *
*******************************************************************************
*/

#ifndef _TIMEZONERULETEST_
#define _TIMEZONERULETEST_

#include "unicode/utypes.h"
#include "caltztst.h"

#if !UCONFIG_NO_FORMATTING

/**
 * Tests for TimeZoneRule, RuleBasedTimeZone and VTimeZone
 */
class TimeZoneRuleTest : public CalendarTimeZoneTest {
    // IntlTest override
    void runIndexedTest(int32_t index, UBool exec, const char*& name, char* par) override;
public:
    void TestSimpleRuleBasedTimeZone();
    void TestHistoricalRuleBasedTimeZone();
    void TestOlsonTransition();
    void TestRBTZTransition();
    void TestHasEquivalentTransitions();
    void TestVTimeZoneRoundTrip();
    void TestVTimeZoneRoundTripPartial();
    void TestVTimeZoneSimpleWrite();
    void TestVTimeZoneHeaderProps();
    void TestGetSimpleRules();
    void TestTimeZoneRuleCoverage();
    void TestSimpleTimeZoneCoverage();
    void TestVTimeZoneCoverage();
    void TestVTimeZoneParse();
    void TestT6216();
    void TestT6669();
    void TestVTimeZoneWrapper();
    void TestT8943();

private:
    void verifyTransitions(BasicTimeZone& icutz, UDate start, UDate end);
    void compareTransitionsAscending(BasicTimeZone& z1, BasicTimeZone& z2,
        UDate start, UDate end, UBool inclusive);
    void compareTransitionsDescending(BasicTimeZone& z1, BasicTimeZone& z2,
        UDate start, UDate end, UBool inclusive);
    UDate getUTCMillis(int32_t year, int32_t month, int32_t dom,
        int32_t hour=0, int32_t min=0, int32_t sec=0, int32_t msec=0);
};

#endif /* #if !UCONFIG_NO_FORMATTING */
 
#endif // _TIMEZONERULETEST_
