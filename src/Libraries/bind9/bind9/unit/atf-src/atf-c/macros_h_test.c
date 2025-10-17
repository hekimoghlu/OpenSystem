/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
#include <atf-c/macros.h>

void atf_require_inside_if(void);
void atf_require_equal_inside_if(void);
void atf_check_errno_semicolons(void);
void atf_require_errno_semicolons(void);

void
atf_require_inside_if(void)
{
    /* Make sure that ATF_REQUIRE can be used inside an if statement that
     * does not have braces.  Earlier versions of it generated an error
     * if there was an else clause because they confused the compiler
     * by defining an unprotected nested if. */
    if (true)
        ATF_REQUIRE(true);
    else
        ATF_REQUIRE(true);
}

void
atf_require_equal_inside_if(void)
{
    /* Make sure that ATF_REQUIRE_EQUAL can be used inside an if statement
     * that does not have braces.  Earlier versions of it generated an
     * error if there was an else clause because they confused the
     * compiler by defining an unprotected nested if. */
    if (true)
        ATF_REQUIRE_EQ(true, true);
    else
        ATF_REQUIRE_EQ(true, true);
}

void
atf_check_errno_semicolons(void)
{
    /* Check that ATF_CHECK_ERRNO does not contain a semicolon that would
     * cause an empty-statement that confuses some compilers. */
    ATF_CHECK_ERRNO(1, 1 == 1);
    ATF_CHECK_ERRNO(2, 2 == 2);
}

void
atf_require_errno_semicolons(void)
{
    /* Check that ATF_REQUIRE_ERRNO does not contain a semicolon that would
     * cause an empty-statement that confuses some compilers. */
    ATF_REQUIRE_ERRNO(1, 1 == 1);
    ATF_REQUIRE_ERRNO(2, 2 == 2);
}

/* Test case names should not be expanded during instatiation so that they
 * can have the exact same name as macros. */
#define TEST_MACRO_1 invalid + name
#define TEST_MACRO_2 invalid + name
#define TEST_MACRO_3 invalid + name
ATF_TC(TEST_MACRO_1);
ATF_TC_HEAD(TEST_MACRO_1, tc) { if (tc != NULL) {} }
ATF_TC_BODY(TEST_MACRO_1, tc) { if (tc != NULL) {} }
atf_tc_t *test_name_1 = &ATF_TC_NAME(TEST_MACRO_1);
void (*head_1)(atf_tc_t *) = ATF_TC_HEAD_NAME(TEST_MACRO_1);
void (*body_1)(const atf_tc_t *) = ATF_TC_BODY_NAME(TEST_MACRO_1);
ATF_TC_WITH_CLEANUP(TEST_MACRO_2);
ATF_TC_HEAD(TEST_MACRO_2, tc) { if (tc != NULL) {} }
ATF_TC_BODY(TEST_MACRO_2, tc) { if (tc != NULL) {} }
ATF_TC_CLEANUP(TEST_MACRO_2, tc) { if (tc != NULL) {} }
atf_tc_t *test_name_2 = &ATF_TC_NAME(TEST_MACRO_2);
void (*head_2)(atf_tc_t *) = ATF_TC_HEAD_NAME(TEST_MACRO_2);
void (*body_2)(const atf_tc_t *) = ATF_TC_BODY_NAME(TEST_MACRO_2);
void (*cleanup_2)(const atf_tc_t *) = ATF_TC_CLEANUP_NAME(TEST_MACRO_2);
ATF_TC_WITHOUT_HEAD(TEST_MACRO_3);
ATF_TC_BODY(TEST_MACRO_3, tc) { if (tc != NULL) {} }
atf_tc_t *test_name_3 = &ATF_TC_NAME(TEST_MACRO_3);
void (*body_3)(const atf_tc_t *) = ATF_TC_BODY_NAME(TEST_MACRO_3);
