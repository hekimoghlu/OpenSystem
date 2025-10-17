/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
 * Copyright (c) 2010-2016, International Business Machines Corporation
 * and others. All Rights Reserved.
 ********************************************************************/

#include "intltest.h"
#include "unicode/locdspnm.h"

/**
 * Tests for the LocaleDisplayNames class
 **/
class LocaleDisplayNamesTest: public IntlTest {
public:
    LocaleDisplayNamesTest();
    virtual ~LocaleDisplayNamesTest();

    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par = nullptr) override;

#if !UCONFIG_NO_FORMATTING
    /**
     * Test methods to set and get data fields
     **/
    void TestCreate();
    void TestCreateDialect();
    void TestWithKeywordsAndEverything();
    void TestUldnOpen();
    void TestUldnOpenDialect();
    void TestUldnWithKeywordsAndEverything();
    void TestUldnComponents();
    void TestRootEtc();
    void TestCurrencyKeyword();
    void TestUnknownCurrencyKeyword();
    void TestUntranslatedKeywords();
    void TestPrivateUse();
    void TestUldnDisplayContext();
    void TestUldnWithGarbage();
    void TestSubstituteHandling();
    void TestNumericRegionID();

    void VerifySubstitute(LocaleDisplayNames* ldn);
    void VerifyNoSubstitute(LocaleDisplayNames* ldn);
#endif

};
