/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#include <string.h>

#include <atf-c.h>

#include "atf-c/config.h"

#include "detail/env.h"
#include "detail/test_helpers.h"

static const char *test_value = "env-value";

static struct varnames {
    const char *lc;
    const char *uc;
    bool can_be_empty;
} all_vars[] = {
    { "atf_arch",           "ATF_ARCH",           false },
    { "atf_build_cc",       "ATF_BUILD_CC",       false },
    { "atf_build_cflags",   "ATF_BUILD_CFLAGS",   true  },
    { "atf_build_cpp",      "ATF_BUILD_CPP",      false },
    { "atf_build_cppflags", "ATF_BUILD_CPPFLAGS", true  },
    { "atf_build_cxx",      "ATF_BUILD_CXX",      false },
    { "atf_build_cxxflags", "ATF_BUILD_CXXFLAGS", true  },
    { "atf_confdir",        "ATF_CONFDIR",        false },
    { "atf_includedir",     "ATF_INCLUDEDIR",     false },
    { "atf_libdir",         "ATF_LIBDIR",         false },
    { "atf_libexecdir",     "ATF_LIBEXECDIR",     false },
    { "atf_machine",        "ATF_MACHINE",        false },
    { "atf_pkgdatadir",     "ATF_PKGDATADIR",     false },
    { "atf_shell",          "ATF_SHELL",          false },
    { "atf_workdir",        "ATF_WORKDIR",        false },
    { NULL,                 NULL,                 false }
};

/* ---------------------------------------------------------------------
 * Auxiliary functions.
 * --------------------------------------------------------------------- */

void __atf_config_reinit(void);

static
void
unset_all(void)
{
    const struct varnames *v;
    for (v = all_vars; v->lc != NULL; v++)
        RE(atf_env_unset(v->uc));
}

static
void
compare_one(const char *var, const char *expvalue)
{
    const struct varnames *v;

    printf("Checking that %s is set to %s\n", var, expvalue);

    for (v = all_vars; v->lc != NULL; v++) {
        if (strcmp(v->lc, var) == 0)
            ATF_CHECK_STREQ(atf_config_get(v->lc), test_value);
        else
            ATF_CHECK(strcmp(atf_config_get(v->lc), test_value) != 0);
    }
}

/* ---------------------------------------------------------------------
 * Test cases for the free functions.
 * --------------------------------------------------------------------- */

ATF_TC(get);
ATF_TC_HEAD(get, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_config_get function");
}
ATF_TC_BODY(get, tc)
{
    const struct varnames *v;

    /* Unset all known environment variables and make sure the built-in
     * values do not match the bogus value we will use for testing. */
    unset_all();
    __atf_config_reinit();
    for (v = all_vars; v->lc != NULL; v++)
        ATF_CHECK(strcmp(atf_config_get(v->lc), test_value) != 0);

    /* Test the behavior of empty values. */
    for (v = all_vars; v->lc != NULL; v++) {
        unset_all();
        if (strcmp(atf_config_get(v->lc), "") != 0) {
            RE(atf_env_set(v->uc, ""));
            __atf_config_reinit();
            if (v->can_be_empty)
                ATF_CHECK(strlen(atf_config_get(v->lc)) == 0);
            else
                ATF_CHECK(strlen(atf_config_get(v->lc)) > 0);
        }
    }

    /* Check if every variable is recognized individually. */
    for (v = all_vars; v->lc != NULL; v++) {
        unset_all();
        RE(atf_env_set(v->uc, test_value));
        __atf_config_reinit();
        compare_one(v->lc, test_value);
    }
}

/* ---------------------------------------------------------------------
 * Tests cases for the header file.
 * --------------------------------------------------------------------- */

HEADER_TC(include, "atf-c/config.h");

/* ---------------------------------------------------------------------
 * Main.
 * --------------------------------------------------------------------- */

ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, get);

    /* Add the test cases for the header file. */
    ATF_TP_ADD_TC(tp, include);

    return atf_no_error();
}
