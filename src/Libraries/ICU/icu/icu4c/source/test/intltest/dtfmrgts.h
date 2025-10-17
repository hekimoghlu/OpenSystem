/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
 * Copyright (c) 1997-2014, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef _DATEFORMATREGRESSIONTEST_
#define _DATEFORMATREGRESSIONTEST_

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/unistr.h"
#include "unicode/smpdtfmt.h" 
#include "caltztst.h"

/** 
 * Performs regression test for DateFormat
 **/
class DateFormatRegressionTest: public CalendarTimeZoneTest {
    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
public:

    void Test4029195();
    void Test4052408();
    void Test4056591();
    void Test4059917();
    void aux917( SimpleDateFormat *fmt, UnicodeString& str );
    void Test4060212();
    void Test4061287();
    void Test4065240();
    void Test4071441();
    void Test4073003();
    void Test4089106();
    void Test4100302();
    void Test4101483();
    void Test4103340();
    void Test4103341();
    void Test4104136();
    void Test4104522();
    void Test4106807();
    void Test4108407(); 
    void Test4134203();
    void Test4151631();
    void Test4151706();
    void Test4162071();
    void Test4182066();
    void Test4210209();
    void Test714();
    void Test1684();
    void Test5554();
    void Test9237();
    void TestParsing();
    void Test12902_yWithGregoCalInThaiLoc();
    void TestT10334();
    void TestT10619();
    void TestT10855();
    void TestT10858();
    void TestT10906();
    void TestT13380();
 };

#endif /* #if !UCONFIG_NO_FORMATTING */
 
#endif // _DATEFORMATREGRESSIONTEST_
//eof
