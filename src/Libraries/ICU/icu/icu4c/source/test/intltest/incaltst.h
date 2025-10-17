/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
 * Copyright (c) 1997-2007, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef __IntlCalendarTest__
#define __IntlCalendarTest__
 
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/calendar.h"
#include "unicode/smpdtfmt.h"
#include "caltztst.h"

class IntlCalendarTest: public CalendarTimeZoneTest {
public:
    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
public:
    void TestTypes();

    void TestGregorian();

    void TestBuddhist();
    void TestBuddhistFormat();
    void TestBug21043Indian();
    void TestBug21044Hebrew();
    void TestBug21045Islamic();
    void TestBug21046IslamicUmalqura();

    void TestTaiwan();

    void TestJapanese();
    void TestJapaneseFormat();
    void TestJapanese3860();
    void TestForceGannenNumbering();
    
    void TestPersian();
    void TestPersianFormat();

    void TestConsistencyGregorian();
    void TestConsistencyCoptic();
    void TestConsistencyEthiopic();
    void TestConsistencyROC();
    void TestConsistencyChinese();
    void TestConsistencyDangi();
    void TestConsistencyBuddhist();
    void TestConsistencyEthiopicAmeteAlem();
    void TestConsistencyHebrew();
    void TestConsistencyIndian();
    void TestConsistencyIslamic();
    void TestConsistencyIslamicCivil();
    void TestConsistencyIslamicRGSA();
    void TestConsistencyIslamicTBLA();
    void TestConsistencyIslamicUmalqura();
    void TestConsistencyPersian();
    void TestConsistencyJapanese();
    void TestIslamicUmalquraCalendarSlow();
    void TestJapaneseLargeEra();

 protected:
    // Test a Gregorian-Like calendar
    void quasiGregorianTest(Calendar& cal, const Locale& gregoLocale, const int32_t *data);
    void simpleTest(const Locale& loc, const UnicodeString& expect, UDate expectDate, UErrorCode& status);
    void checkConsistency(const char* locale);

    int32_t daysToCheckInConsistency;
 
public: // package
    // internal routine for checking date
    static UnicodeString value(Calendar* calendar);
 
};


#endif /* #if !UCONFIG_NO_FORMATTING */
 
#endif // __IntlCalendarTest__
