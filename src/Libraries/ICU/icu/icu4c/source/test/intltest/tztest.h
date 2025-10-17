/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
 * Copyright (c) 1997-2014, International Business Machines
 * Corporation and others. All Rights Reserved.
 ********************************************************************/
 
#ifndef __TimeZoneTest__
#define __TimeZoneTest__

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/simpletz.h" 
#include "caltztst.h"

/** 
 * Various tests for TimeZone
 **/
class TimeZoneTest: public CalendarTimeZoneTest {
    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
public: // package
    static const int32_t millisPerHour;
 
public:
    /**
     * Test the offset of the PRT timezone.
     */
    virtual void TestPRTOffset();
    /**
     * Regress a specific bug with a sequence of API calls.
     */
    virtual void TestVariousAPI518();
    /**
     * Test the call which retrieves the available IDs.
     */
    virtual void TestGetAvailableIDs913();

    virtual void TestGetAvailableIDsNew();

    /**
     * Generic API testing for API coverage.
     */
    virtual void TestGenericAPI();
    /**
     * Test the setStartRule/setEndRule API calls.
     */
    virtual void TestRuleAPI();
 
    void findTransition(const TimeZone& tz,
                        UDate min, UDate max);

   /**
     * subtest used by TestRuleAPI
     **/
    void testUsingBinarySearch(const TimeZone& tz,
                               UDate min, UDate max,
                               UDate expectedBoundary);


    /**
     *  Test short zone IDs for compliance
     */ 
    virtual void TestShortZoneIDs();


    /**
     *  Test parsing custom zones
     */ 
    virtual void TestCustomParse();
    
    /**
     *  Test new getDisplayName() API
     */ 
    virtual void TestDisplayName();

    void TestDSTSavings();
    void TestAlternateRules();

    void TestCountries();

    void TestHistorical();

    void TestEquivalentIDs();

    void TestAliasedNames();
    
    void TestFractionalDST();

    void TestFebruary();

    void TestCanonicalIDAPI();
    void TestCanonicalID();

    virtual void TestDisplayNamesMeta();

    void TestGetRegion();
    void TestGetUnknown();
    void TestGetGMT();

    void TestGetWindowsID();
    void TestGetIDForWindowsID();
    void TestCasablancaNameAndOffset22041();
    void TestRawOffsetAndOffsetConsistency22041();
    void TestGMTMinus24ICU22526();

    void TestGetIanaID();

    static const UDate INTERVAL;

private:
    // internal functions
    static UnicodeString& formatOffset(int32_t offset, UnicodeString& rv);
    static UnicodeString& formatTZID(int32_t offset, UnicodeString& rv);

    // Some test case data is current date/tzdata version sensitive and producing errors
    // when year/rule are changed.
    static const int32_t REFERENCE_YEAR;
    static const char *REFERENCE_DATA_VERSION;

    void checkContainsAll(StringEnumeration *s1, const char *name1,
        StringEnumeration *s2, const char *name2);
};

#endif /* #if !UCONFIG_NO_FORMATTING */
 
#endif // __TimeZoneTest__
