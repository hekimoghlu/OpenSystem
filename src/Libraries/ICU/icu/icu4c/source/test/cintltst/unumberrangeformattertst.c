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

// Â© 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

// Allow implicit conversion from char16_t* to UnicodeString for this file:
// Helpful in toString methods and elsewhere.
#define UNISTR_FROM_STRING_EXPLICIT

#include <stdbool.h>
#include <stdio.h>
#include "unicode/unumberformatter.h"
#include "unicode/unumberrangeformatter.h"
#include "unicode/umisc.h"
#include "unicode/unum.h"
#include "unicode/ustring.h"
#include "cformtst.h"
#include "cintltst.h"
#include "cmemory.h"

static void TestExampleCode(void);

static void TestFormattedValue(void);

static void TestSkeletonParseError(void);

static void TestGetDecimalNumbers(void);

void addUNumberRangeFormatterTest(TestNode** root);

#define TESTCASE(x) addTest(root, &x, "tsformat/unumberrangeformatter/" #x)

void addUNumberRangeFormatterTest(TestNode** root) {
    TESTCASE(TestExampleCode);
    TESTCASE(TestFormattedValue);
    TESTCASE(TestSkeletonParseError);
    TESTCASE(TestGetDecimalNumbers);
}


#define CAPACITY 30


static void TestExampleCode(void) {
    // This is the example code given in unumberrangeformatter.h.

    // Setup:
    UErrorCode ec = U_ZERO_ERROR;
    UNumberRangeFormatter* uformatter = unumrf_openForSkeletonWithCollapseAndIdentityFallback(
        u"currency/USD precision-integer",
        -1,
        UNUM_RANGE_COLLAPSE_AUTO,
        UNUM_IDENTITY_FALLBACK_APPROXIMATELY,
        "en-US",
        NULL,
        &ec);
    UFormattedNumberRange* uresult = unumrf_openResult(&ec);
    assertSuccessCheck("There should not be a failure in the example code", &ec, true);

    // Format a double range:
    unumrf_formatDoubleRange(uformatter, 3.0, 5.0, uresult, &ec);
    assertSuccessCheck("There should not be a failure in the example code", &ec, true);

    // Get the result string:
    int32_t len;
    const UChar* str = ufmtval_getString(unumrf_resultAsValue(uresult, &ec), &len, &ec);
    assertSuccessCheck("There should not be a failure in the example code", &ec, true);
    assertUEquals("Should produce expected string result", u"$3 â€“ $5", str);
    int32_t resultLength = str != NULL ? u_strlen(str) : 0;
    assertIntEquals("Length should be as expected", resultLength, len);

    // Cleanup:
    unumrf_close(uformatter);
    unumrf_closeResult(uresult);
}


static void TestFormattedValue(void) {
    UErrorCode ec = U_ZERO_ERROR;
    UNumberRangeFormatter* uformatter = unumrf_openForSkeletonWithCollapseAndIdentityFallback(
        u"K",
        -1,
        UNUM_RANGE_COLLAPSE_AUTO,
        UNUM_IDENTITY_FALLBACK_APPROXIMATELY,
        "en-US",
        NULL,
        &ec);
    assertSuccessCheck("Should create without error", &ec, true);
    UFormattedNumberRange* uresult = unumrf_openResult(&ec);
    assertSuccess("Should create result without error", &ec);

    // Test the decimal number code path, too
    unumrf_formatDecimalRange(uformatter, "5.5e4", -1, "1.5e5", -1, uresult, &ec);

    if (assertSuccessCheck("Should format without error", &ec, true)) {
        const UFormattedValue* fv = unumrf_resultAsValue(uresult, &ec);
        assertSuccess("Should convert without error", &ec);
        static const UFieldPositionWithCategory expectedFieldPositions[] = {
            // category, field, begin index, end index
            {UFIELD_CATEGORY_NUMBER_RANGE_SPAN, 0, 0, 3},
            {UFIELD_CATEGORY_NUMBER, UNUM_INTEGER_FIELD, 0, 2},
            {UFIELD_CATEGORY_NUMBER, UNUM_COMPACT_FIELD, 2, 3},
            {UFIELD_CATEGORY_NUMBER_RANGE_SPAN, 1, 6, 10},
            {UFIELD_CATEGORY_NUMBER, UNUM_INTEGER_FIELD, 6, 9},
            {UFIELD_CATEGORY_NUMBER, UNUM_COMPACT_FIELD, 9, 10}};
        checkMixedFormattedValue(
            "FormattedNumber as FormattedValue",
            fv,
            u"55K â€“ 150K",
            expectedFieldPositions,
            UPRV_LENGTHOF(expectedFieldPositions));
    }

    assertIntEquals("Identity result should match",
        UNUM_IDENTITY_RESULT_NOT_EQUAL,
        unumrf_resultGetIdentityResult(uresult, &ec));

    // cleanup:
    unumrf_closeResult(uresult);
    unumrf_close(uformatter);
}


static void TestSkeletonParseError(void) {
    UErrorCode ec = U_ZERO_ERROR;
    UNumberRangeFormatter* uformatter;
    UParseError perror;

    // The UParseError can be null. The following should not segfault.
    uformatter = unumrf_openForSkeletonWithCollapseAndIdentityFallback(
        u".00 measure-unit/typo",
        -1, 
        UNUM_RANGE_COLLAPSE_AUTO,
        UNUM_IDENTITY_FALLBACK_APPROXIMATELY,
        "en",
        NULL,
        &ec);
    unumrf_close(uformatter);

    // Now test the behavior.
    ec = U_ZERO_ERROR;
    uformatter = unumrf_openForSkeletonWithCollapseAndIdentityFallback(
        u".00 measure-unit/typo",
        -1, 
        UNUM_RANGE_COLLAPSE_AUTO,
        UNUM_IDENTITY_FALLBACK_APPROXIMATELY,
        "en",
        &perror,
        &ec);

    assertIntEquals("Should have set error code", U_NUMBER_SKELETON_SYNTAX_ERROR, ec);
    assertIntEquals("Should have correct skeleton error offset", 17, perror.offset);
    assertUEquals("Should have correct pre context", u"0 measure-unit/", perror.preContext);
    assertUEquals("Should have correct post context", u"typo", perror.postContext);

    // cleanup:
    unumrf_close(uformatter);
}


static void TestGetDecimalNumbers(void) {
    UErrorCode ec = U_ZERO_ERROR;
    UNumberRangeFormatter* uformatter = unumrf_openForSkeletonWithCollapseAndIdentityFallback(
        u"currency/USD",
        -1,
        UNUM_RANGE_COLLAPSE_AUTO,
        UNUM_IDENTITY_FALLBACK_APPROXIMATELY,
        "en-US",
        NULL,
        &ec);
    assertSuccessCheck("Should create without error", &ec, true);
    UFormattedNumberRange* uresult = unumrf_openResult(&ec);
    assertSuccess("Should create result without error", &ec);

    unumrf_formatDoubleRange(uformatter, 3.0, 5.0, uresult, &ec);
    const UChar* str = ufmtval_getString(unumrf_resultAsValue(uresult, &ec), NULL, &ec);
    assertSuccessCheck("Formatting should succeed", &ec, true);
    assertUEquals("Should produce expected string result", u"$3.00 \u2013 $5.00", str);

    char buffer[CAPACITY];

    int32_t len = unumrf_resultGetFirstDecimalNumber(uresult, buffer, CAPACITY, &ec);
    assertIntEquals("First len should be as expected", strlen(buffer), len);
    assertEquals("First decimal should be as expected", "3", buffer);

    len = unumrf_resultGetSecondDecimalNumber(uresult, buffer, CAPACITY, &ec);
    assertIntEquals("Second len should be as expected", strlen(buffer), len);
    assertEquals("Second decimal should be as expected", "5", buffer);

    // cleanup:
    unumrf_closeResult(uresult);
    unumrf_close(uformatter);
}

#endif /* #if !UCONFIG_NO_FORMATTING */
