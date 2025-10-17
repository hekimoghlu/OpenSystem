/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
 * Copyright (c) 2001, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/************************************************************************
*   This test program is intended for testing Replaceable class.
*
*   Date        Name        Description
*   11/28/2001  hshih       Creation.
* 
************************************************************************/


#ifndef REPTEST_H
#define REPTEST_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/translit.h"
#include "intltest.h"

/**
 * @test
 * @summary Testing the Replaceable class
 */
class ReplaceableTest : public IntlTest {
public:
    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par=nullptr) override;

    /*Tests the Replaceable class according to the API documentation. */
    void TestReplaceableClass();
private:
    void check( const UnicodeString& transliteratorName, 
                const UnicodeString& test, 
                const UnicodeString& shouldProduceStyles);
};

#endif /* #if !UCONFIG_NO_TRANSLITERATION */

#endif
