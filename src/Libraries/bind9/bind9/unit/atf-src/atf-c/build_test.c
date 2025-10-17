/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atf-c.h>

#include "atf-c/build.h"
#include "atf-c/config.h"
#include "atf-c/utils.h"

#include "detail/env.h"
#include "detail/test_helpers.h"
#include "h_build.h"

/* ---------------------------------------------------------------------
 * Auxiliary functions.
 * --------------------------------------------------------------------- */

void __atf_config_reinit(void);

static
bool
equal_arrays(const char *const *exp_array, char **actual_array)
{
    bool equal;

    if (*exp_array == NULL && *actual_array == NULL)
        equal = true;
    else if (*exp_array == NULL || *actual_array == NULL)
        equal = false;
    else {
        equal = true;
        while (*actual_array != NULL) {
            if (*exp_array == NULL || strcmp(*exp_array, *actual_array) != 0) {
                equal = false;
                break;
            }
            exp_array++;
            actual_array++;
        }
    }

    return equal;
}

static
void
check_equal_array(const char *const *exp_array, char **actual_array)
{
    {
        const char *const *exp_ptr;
        printf("Expected arguments:");
        for (exp_ptr = exp_array; *exp_ptr != NULL; exp_ptr++)
            printf(" '%s'", *exp_ptr);
        printf("\n");
    }

    {
        char **actual_ptr;
        printf("Returned arguments:");
        for (actual_ptr = actual_array; *actual_ptr != NULL; actual_ptr++)
            printf(" '%s'", *actual_ptr);
        printf("\n");
    }

    if (!equal_arrays(exp_array, actual_array))
        atf_tc_fail_nonfatal("The constructed argv differs from the "
                             "expected values");
}

static
void
verbose_set_env(const char *var, const char *val)
{
    printf("Setting %s to '%s'\n", var, val);
    RE(atf_env_set(var, val));
}

/* ---------------------------------------------------------------------
 * Internal test cases.
 * --------------------------------------------------------------------- */

ATF_TC(equal_arrays);
ATF_TC_HEAD(equal_arrays, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the test case internal "
                      "equal_arrays function");
}
ATF_TC_BODY(equal_arrays, tc)
{
    {
        const char *const exp[] = { NULL };
        char *actual[] = { NULL };

        ATF_CHECK(equal_arrays(exp, actual));
    }

    {
        const char *const exp[] = { NULL };
        char *actual[2] = { strdup("foo"), NULL };

        ATF_CHECK(!equal_arrays(exp, actual));
        free(actual[0]);
    }

    {
        const char *const exp[] = { "foo", NULL };
        char *actual[] = { NULL };

        ATF_CHECK(!equal_arrays(exp, actual));
    }

    {
        const char *const exp[] = { "foo", NULL };
        char *actual[2] = { strdup("foo"), NULL };

        ATF_CHECK(equal_arrays(exp, actual));
        free(actual[0]);
    }
}

/* ---------------------------------------------------------------------
 * Test cases for the free functions.
 * --------------------------------------------------------------------- */

ATF_TC(c_o);
ATF_TC_HEAD(c_o, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_build_c_o function");
}
ATF_TC_BODY(c_o, tc)
{
    struct c_o_test *test;

    for (test = c_o_tests; test->expargv[0] != NULL; test++) {
        printf("> Test: %s\n", test->msg);

        verbose_set_env("ATF_BUILD_CC", test->cc);
        verbose_set_env("ATF_BUILD_CFLAGS", test->cflags);
        verbose_set_env("ATF_BUILD_CPPFLAGS", test->cppflags);
        __atf_config_reinit();

        {
            char **argv;
            if (test->hasoptargs)
                RE(atf_build_c_o(test->sfile, test->ofile, test->optargs,
                                 &argv));
            else
                RE(atf_build_c_o(test->sfile, test->ofile, NULL, &argv));
            check_equal_array(test->expargv, argv);
            atf_utils_free_charpp(argv);
        }
    }
}

ATF_TC(cpp);
ATF_TC_HEAD(cpp, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_build_cpp function");
}
ATF_TC_BODY(cpp, tc)
{
    struct cpp_test *test;

    for (test = cpp_tests; test->expargv[0] != NULL; test++) {
        printf("> Test: %s\n", test->msg);

        verbose_set_env("ATF_BUILD_CPP", test->cpp);
        verbose_set_env("ATF_BUILD_CPPFLAGS", test->cppflags);
        __atf_config_reinit();

        {
            char **argv;
            if (test->hasoptargs)
                RE(atf_build_cpp(test->sfile, test->ofile, test->optargs,
                                 &argv));
            else
                RE(atf_build_cpp(test->sfile, test->ofile, NULL, &argv));
            check_equal_array(test->expargv, argv);
            atf_utils_free_charpp(argv);
        }
    }
}

ATF_TC(cxx_o);
ATF_TC_HEAD(cxx_o, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_build_cxx_o function");
}
ATF_TC_BODY(cxx_o, tc)
{
    struct cxx_o_test *test;

    for (test = cxx_o_tests; test->expargv[0] != NULL; test++) {
        printf("> Test: %s\n", test->msg);

        verbose_set_env("ATF_BUILD_CXX", test->cxx);
        verbose_set_env("ATF_BUILD_CXXFLAGS", test->cxxflags);
        verbose_set_env("ATF_BUILD_CPPFLAGS", test->cppflags);
        __atf_config_reinit();

        {
            char **argv;
            if (test->hasoptargs)
                RE(atf_build_cxx_o(test->sfile, test->ofile, test->optargs,
                                   &argv));
            else
                RE(atf_build_cxx_o(test->sfile, test->ofile, NULL, &argv));
            check_equal_array(test->expargv, argv);
            atf_utils_free_charpp(argv);
        }
    }
}

/* ---------------------------------------------------------------------
 * Tests cases for the header file.
 * --------------------------------------------------------------------- */

HEADER_TC(include, "atf-c/build.h");

/* ---------------------------------------------------------------------
 * Main.
 * --------------------------------------------------------------------- */

ATF_TP_ADD_TCS(tp)
{
    /* Add the internal test cases. */
    ATF_TP_ADD_TC(tp, equal_arrays);

    /* Add the test cases for the free functions. */
    ATF_TP_ADD_TC(tp, c_o);
    ATF_TP_ADD_TC(tp, cpp);
    ATF_TP_ADD_TC(tp, cxx_o);

    /* Add the test cases for the header file. */
    ATF_TP_ADD_TC(tp, include);

    return atf_no_error();
}
