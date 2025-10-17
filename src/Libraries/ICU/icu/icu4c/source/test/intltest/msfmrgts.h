/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#ifndef _MESSAGEFORMATREGRESSIONTEST_
#define _MESSAGEFORMATREGRESSIONTEST_

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"

/**
 * Performs regression test for MessageFormat
 **/
class MessageFormatRegressionTest: public IntlTest {

    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
public:

    void Test4074764();
    void Test4058973();
    void Test4031438();
    void Test4052223();
    void Test4104976();
    void Test4106659();
    void Test4106660();
    void Test4111739();
    void Test4114743();
    void Test4116444();
    void Test4114739();
    void Test4113018();
    void Test4106661();
    void Test4094906();
    void Test4118592();
    void Test4118594();
    void Test4105380();
    void Test4120552();
    void Test4142938();
    void TestChoicePatternQuote();
    void Test4112104();
    void TestICU12584();
    void TestICU22798();
    void TestAPI();

protected:
    UBool failure(UErrorCode status, const char* msg, UBool possibleDataError=false);

};

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif // _MESSAGEFORMATREGRESSIONTEST_
//eof
