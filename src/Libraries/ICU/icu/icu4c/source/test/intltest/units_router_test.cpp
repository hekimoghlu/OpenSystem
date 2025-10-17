/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

// Â© 2020 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html#License

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"
#include "unicode/unistr.h"
#include "units_router.h"


class UnitsRouterTest : public IntlTest {
  public:
    UnitsRouterTest() {}

    void runIndexedTest(int32_t index, UBool exec, const char *&name, char *par = nullptr) override;

    void testBasic();
};

extern IntlTest *createUnitsRouterTest() { return new UnitsRouterTest(); }

void UnitsRouterTest::runIndexedTest(int32_t index, UBool exec, const char *&name, char * /*par*/) {
    if (exec) { logln("TestSuite UnitsRouterTest: "); }
    TESTCASE_AUTO_BEGIN;
    TESTCASE_AUTO(testBasic);
    TESTCASE_AUTO_END;
}

void UnitsRouterTest::testBasic() { IcuTestErrorCode status(*this, "UnitsRouter testBasic"); }

#endif /* #if !UCONFIG_NO_FORMATTING */
