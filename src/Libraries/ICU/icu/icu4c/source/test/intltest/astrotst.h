/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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

#ifndef __AstroTest__
#define __AstroTest__
 
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/calendar.h"
#include "unicode/gregocal.h"
#include "unicode/smpdtfmt.h"
#include "astro.h"
#include "caltztst.h"

class AstroTest: public CalendarTimeZoneTest {
public:
    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
public:
    AstroTest();

    void TestSolarLongitude();

    void TestLunarPosition();

    void TestCoordinates();

    void TestCoverage();

    void TestBasics();
    
    void TestMoonAge();
 private:
    void init(UErrorCode&);
    void close(UErrorCode&);
    
    Calendar *gc;
    
};

#endif /* #if !UCONFIG_NO_FORMATTING */
 
#endif // __AstroTest__
