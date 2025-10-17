/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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

// Â© 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "intltest.h"
#include "unicode/localebuilder.h"


/**
 * Tests for the LocaleBuilder class
 **/
class LocaleBuilderTest: public IntlTest {
 public:
    LocaleBuilderTest();
    virtual ~LocaleBuilderTest();

    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    void TestAddRemoveUnicodeLocaleAttribute();
    void TestAddRemoveUnicodeLocaleAttributeWellFormed();
    void TestAddUnicodeLocaleAttributeIllFormed();
    void TestLocaleBuilder();
    void TestLocaleBuilderBasic();
    void TestLocaleBuilderBasicWithExtensionsOnDefaultLocale();
    void TestPosixCases();
    void TestSetExtensionOthers();
    void TestSetExtensionPU();
    void TestSetExtensionT();
    void TestSetExtensionU();
    void TestSetExtensionValidateOthersIllFormed();
    void TestSetExtensionValidateOthersWellFormed();
    void TestSetExtensionValidatePUIllFormed();
    void TestSetExtensionValidatePUWellFormed();
    void TestSetExtensionValidateTIllFormed();
    void TestSetExtensionValidateTWellFormed();
    void TestSetExtensionValidateUIllFormed();
    void TestSetExtensionValidateUWellFormed();
    void TestSetLanguageIllFormed();
    void TestSetLanguageWellFormed();
    void TestSetLocale();
    void TestSetRegionIllFormed();
    void TestSetRegionWellFormed();
    void TestSetScriptIllFormed();
    void TestSetScriptWellFormed();
    void TestSetUnicodeLocaleKeywordIllFormedKey();
    void TestSetUnicodeLocaleKeywordIllFormedValue();
    void TestSetUnicodeLocaleKeywordWellFormed();
    void TestSetVariantIllFormed();
    void TestSetVariantWellFormed();

 private:
    void Verify(LocaleBuilder& bld, const char* expected, const char* msg);
};
