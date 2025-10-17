/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
 * Copyright (c) 1997-2011, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/


#ifndef _PUTILTEST_
#define _PUTILTEST_

#include "intltest.h"

/** 
 * Test putil.h
 **/
class PUtilTest : public IntlTest {
    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
public:

//    void testIEEEremainder();
    void testMaxMin();

private:
//    void remainderTest(double x, double y, double exp);
    void maxMinTest(double a, double b, double exp, UBool max);

    // the actual tests; methods return the number of errors
    void testNaN();
    void testPositiveInfinity();
    void testNegativeInfinity();
    void testZero();

    // subtests of testNaN
    void testIsNaN();
    void NaNGT();
    void NaNLT();
    void NaNGTE();
    void NaNLTE();
    void NaNE();
    void NaNNE();

};
 
#endif
//eof
