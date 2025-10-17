/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#include <stdlib.h>
#include <string.h>

#include <atf-c.h>

#include "env.h"
#include "test_helpers.h"
#include "text.h"

/* ---------------------------------------------------------------------
 * Test cases for the free functions.
 * --------------------------------------------------------------------- */

ATF_TC(has);
ATF_TC_HEAD(has, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_env_has function");
}
ATF_TC_BODY(has, tc)
{
    ATF_REQUIRE(atf_env_has("PATH"));
    ATF_REQUIRE(!atf_env_has("_UNDEFINED_VARIABLE_"));
}

ATF_TC(get);
ATF_TC_HEAD(get, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_env_get function");
}
ATF_TC_BODY(get, tc)
{
    const char *val;

    ATF_REQUIRE(atf_env_has("PATH"));

    val = atf_env_get("PATH");
    ATF_REQUIRE(strlen(val) > 0);
    ATF_REQUIRE(strchr(val, ':') != NULL);
}

ATF_TC(set);
ATF_TC_HEAD(set, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_env_set function");
}
ATF_TC_BODY(set, tc)
{
    char *oldval;

    ATF_REQUIRE(atf_env_has("PATH"));
    RE(atf_text_format(&oldval, "%s", atf_env_get("PATH")));
    RE(atf_env_set("PATH", "foo-bar"));
    ATF_REQUIRE(strcmp(atf_env_get("PATH"), oldval) != 0);
    ATF_REQUIRE(strcmp(atf_env_get("PATH"), "foo-bar") == 0);
    free(oldval);

    ATF_REQUIRE(!atf_env_has("_UNDEFINED_VARIABLE_"));
    RE(atf_env_set("_UNDEFINED_VARIABLE_", "foo2-bar2"));
    ATF_REQUIRE(strcmp(atf_env_get("_UNDEFINED_VARIABLE_"),
                     "foo2-bar2") == 0);
}

ATF_TC(unset);
ATF_TC_HEAD(unset, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_env_unset function");
}
ATF_TC_BODY(unset, tc)
{
    ATF_REQUIRE(atf_env_has("PATH"));
    RE(atf_env_unset("PATH"));
    ATF_REQUIRE(!atf_env_has("PATH"));
}

/* ---------------------------------------------------------------------
 * Main.
 * --------------------------------------------------------------------- */

ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, has);
    ATF_TP_ADD_TC(tp, get);
    ATF_TP_ADD_TC(tp, set);
    ATF_TP_ADD_TC(tp, unset);

    return atf_no_error();
}
