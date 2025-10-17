/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#include <stdbool.h>
#include <string.h>

#include <atf-c.h>

#include "detail/test_helpers.h"

/* ---------------------------------------------------------------------
 * Auxiliary test cases.
 * --------------------------------------------------------------------- */

ATF_TC_HEAD(empty, tc)
{
    if (tc != NULL) {}
}
ATF_TC_BODY(empty, tc)
{
}

ATF_TC_HEAD(test_var, tc)
{
    atf_tc_set_md_var(tc, "test-var", "Test text");
}

/* ---------------------------------------------------------------------
 * Test cases for the "atf_tc_t" type.
 * --------------------------------------------------------------------- */

ATF_TC(init);
ATF_TC_HEAD(init, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_tc_init function");
}
ATF_TC_BODY(init, tcin)
{
    atf_tc_t tc;

    RE(atf_tc_init(&tc, "test1", ATF_TC_HEAD_NAME(empty),
                   ATF_TC_BODY_NAME(empty), NULL, NULL));
    ATF_REQUIRE(strcmp(atf_tc_get_ident(&tc), "test1") == 0);
    ATF_REQUIRE(!atf_tc_has_md_var(&tc, "test-var"));
    atf_tc_fini(&tc);

    RE(atf_tc_init(&tc, "test2", ATF_TC_HEAD_NAME(test_var),
                   ATF_TC_BODY_NAME(empty), NULL, NULL));
    ATF_REQUIRE(strcmp(atf_tc_get_ident(&tc), "test2") == 0);
    ATF_REQUIRE(atf_tc_has_md_var(&tc, "test-var"));
    atf_tc_fini(&tc);
}

ATF_TC(init_pack);
ATF_TC_HEAD(init_pack, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_tc_init_pack function");
}
ATF_TC_BODY(init_pack, tcin)
{
    atf_tc_t tc;
    atf_tc_pack_t tcp1 = {
        .m_ident = "test1",
        .m_head = ATF_TC_HEAD_NAME(empty),
        .m_body = ATF_TC_BODY_NAME(empty),
        .m_cleanup = NULL,
    };
    atf_tc_pack_t tcp2 = {
        .m_ident = "test2",
        .m_head = ATF_TC_HEAD_NAME(test_var),
        .m_body = ATF_TC_BODY_NAME(empty),
        .m_cleanup = NULL,
    };

    RE(atf_tc_init_pack(&tc, &tcp1, NULL));
    ATF_REQUIRE(strcmp(atf_tc_get_ident(&tc), "test1") == 0);
    ATF_REQUIRE(!atf_tc_has_md_var(&tc, "test-var"));
    atf_tc_fini(&tc);

    RE(atf_tc_init_pack(&tc, &tcp2, NULL));
    ATF_REQUIRE(strcmp(atf_tc_get_ident(&tc), "test2") == 0);
    ATF_REQUIRE(atf_tc_has_md_var(&tc, "test-var"));
    atf_tc_fini(&tc);
}

ATF_TC(vars);
ATF_TC_HEAD(vars, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_tc_get_md_var, "
                      "atf_tc_has_md_var and atf_tc_set_md_var functions");
}
ATF_TC_BODY(vars, tcin)
{
    atf_tc_t tc;

    RE(atf_tc_init(&tc, "test1", ATF_TC_HEAD_NAME(empty),
                   ATF_TC_BODY_NAME(empty), NULL, NULL));
    ATF_REQUIRE(!atf_tc_has_md_var(&tc, "test-var"));
    RE(atf_tc_set_md_var(&tc, "test-var", "Test value"));
    ATF_REQUIRE(atf_tc_has_md_var(&tc, "test-var"));
    ATF_REQUIRE(strcmp(atf_tc_get_md_var(&tc, "test-var"), "Test value") == 0);
    atf_tc_fini(&tc);
}

ATF_TC(config);
ATF_TC_HEAD(config, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_tc_get_config_var, "
                      "atf_tc_get_config_var_wd and atf_tc_has_config_var "
                      "functions");
}
ATF_TC_BODY(config, tcin)
{
    atf_tc_t tc;
    const char *const config[] = { "test-var", "test-value", NULL };

    RE(atf_tc_init(&tc, "test1", ATF_TC_HEAD_NAME(empty),
                   ATF_TC_BODY_NAME(empty), NULL, NULL));
    ATF_REQUIRE(!atf_tc_has_config_var(&tc, "test-var"));
    ATF_REQUIRE(!atf_tc_has_md_var(&tc, "test-var"));
    atf_tc_fini(&tc);

    RE(atf_tc_init(&tc, "test1", ATF_TC_HEAD_NAME(empty),
                   ATF_TC_BODY_NAME(empty), NULL, config));
    ATF_REQUIRE(atf_tc_has_config_var(&tc, "test-var"));
    ATF_REQUIRE(strcmp(atf_tc_get_config_var(&tc, "test-var"),
                     "test-value") == 0);
    ATF_REQUIRE(!atf_tc_has_md_var(&tc, "test-var"));
    ATF_REQUIRE(!atf_tc_has_config_var(&tc, "test-var2"));
    ATF_REQUIRE(strcmp(atf_tc_get_config_var_wd(&tc, "test-var2", "def-value"),
                     "def-value") == 0);
    atf_tc_fini(&tc);
}

/* ---------------------------------------------------------------------
 * Test cases for the free functions.
 * --------------------------------------------------------------------- */

/* TODO: Add test cases for atf_tc_run.  This is going to be very tough,
 * but good tests here could allow us to avoid much of the indirect
 * testing done later on. */

/* ---------------------------------------------------------------------
 * Tests cases for the header file.
 * --------------------------------------------------------------------- */

HEADER_TC(include, "atf-c/tc.h");

/* ---------------------------------------------------------------------
 * Main.
 * --------------------------------------------------------------------- */

ATF_TP_ADD_TCS(tp)
{
    /* Add the test cases for the "atf_tcr_t" type. */
    ATF_TP_ADD_TC(tp, init);
    ATF_TP_ADD_TC(tp, init_pack);
    ATF_TP_ADD_TC(tp, vars);
    ATF_TP_ADD_TC(tp, config);

    /* Add the test cases for the free functions. */
    /* TODO */

    /* Add the test cases for the header file. */
    ATF_TP_ADD_TC(tp, include);

    return atf_no_error();
}
