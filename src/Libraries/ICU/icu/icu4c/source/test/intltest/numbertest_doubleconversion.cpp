/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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

// Â© 2017 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "numbertest.h"
#include "double-conversion.h"

using namespace double_conversion;

void DoubleConversionTest::runIndexedTest(int32_t index, UBool exec, const char *&name, char *) {
    if (exec) {
        logln("TestSuite DoubleConversionTest: ");
    }
    TESTCASE_AUTO_BEGIN;
        TESTCASE_AUTO(testDoubleConversionApi);
    TESTCASE_AUTO_END;
}

void DoubleConversionTest::testDoubleConversionApi() {
    double v = 87.65;
    char buffer[DoubleToStringConverter::kBase10MaximalLength + 1];
    bool sign;
    int32_t length;
    int32_t point;

    DoubleToStringConverter::DoubleToAscii(
        v,
        DoubleToStringConverter::DtoaMode::SHORTEST,
        0,
        buffer,
        sizeof(buffer),
        &sign,
        &length,
        &point
    );

    UnicodeString result(buffer, length);
    assertEquals("Digits", u"8765", result);
    assertEquals("Scale", 2, point);
}

#endif /* #if !UCONFIG_NO_FORMATTING */
