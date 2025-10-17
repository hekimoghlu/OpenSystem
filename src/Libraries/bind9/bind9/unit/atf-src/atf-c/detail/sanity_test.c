/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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
#if defined(HAVE_CONFIG_H)
#include "bconfig.h"
#endif

#include <sys/types.h>
#include <sys/wait.h>

#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atf-c.h>

#include "dynstr.h"
#include "process.h"
#include "sanity.h"
#include "test_helpers.h"

/* ---------------------------------------------------------------------
 * Auxiliary functions.
 * --------------------------------------------------------------------- */

enum type { inv, pre, post, unreachable };

struct test_data {
    enum type m_type;
    bool m_cond;
};

static void do_test_child(void *) ATF_DEFS_ATTRIBUTE_NORETURN;

static
void
do_test_child(void *v)
{
    struct test_data *td = v;

    switch (td->m_type) {
    case inv:
        INV(td->m_cond);
        break;

    case pre:
        PRE(td->m_cond);
        break;

    case post:
        POST(td->m_cond);
        break;

    case unreachable:
        if (!td->m_cond)
            UNREACHABLE;
        break;
    }

    exit(EXIT_SUCCESS);
}

static
void
do_test(enum type t, bool cond)
{
    atf_process_child_t child;
    atf_process_status_t status;
    int nlines;
    char *lines[3];

    {
        atf_process_stream_t outsb, errsb;
        struct test_data td = { t, cond };

        RE(atf_process_stream_init_inherit(&outsb));
        RE(atf_process_stream_init_capture(&errsb));
        RE(atf_process_fork(&child, do_test_child, &outsb, &errsb, &td));
        atf_process_stream_fini(&errsb);
        atf_process_stream_fini(&outsb);
    }

    nlines = 0;
    while (nlines < 3 && (lines[nlines] =
           atf_utils_readline(atf_process_child_stderr(&child))) != NULL)
        nlines++;
    ATF_REQUIRE(nlines == 0 || nlines == 3);

    RE(atf_process_child_wait(&child, &status));
    if (!cond) {
        ATF_REQUIRE(atf_process_status_signaled(&status));
        ATF_REQUIRE(atf_process_status_termsig(&status) == SIGABRT);
    } else {
        ATF_REQUIRE(atf_process_status_exited(&status));
        ATF_REQUIRE(atf_process_status_exitstatus(&status) == EXIT_SUCCESS);
    }
    atf_process_status_fini(&status);

    if (!cond) {
        switch (t) {
        case inv:
            ATF_REQUIRE(atf_utils_grep_string("Invariant", lines[0]));
            break;

        case pre:
            ATF_REQUIRE(atf_utils_grep_string("Precondition", lines[0]));
            break;

        case post:
            ATF_REQUIRE(atf_utils_grep_string("Postcondition", lines[0]));
            break;

        case unreachable:
            ATF_REQUIRE(atf_utils_grep_string("Invariant", lines[0]));
            break;
        }

        ATF_REQUIRE(atf_utils_grep_string(__FILE__, lines[0]));
        ATF_REQUIRE(atf_utils_grep_string(PACKAGE_BUGREPORT, lines[2]));
    }

    while (nlines > 0) {
        nlines--;
        free(lines[nlines]);
    }
}

static
void
require_ndebug(void)
{
#if defined(NDEBUG)
    atf_tc_skip("Sanity checks not available; code built with -DNDEBUG");
#endif
}

/* ---------------------------------------------------------------------
 * Test cases for the free functions.
 * --------------------------------------------------------------------- */

ATF_TC(inv);
ATF_TC_HEAD(inv, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the INV macro");
}
ATF_TC_BODY(inv, tc)
{
    require_ndebug();

    do_test(inv, false);
    do_test(inv, true);
}

ATF_TC(pre);
ATF_TC_HEAD(pre, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the PRE macro");
}
ATF_TC_BODY(pre, tc)
{
    require_ndebug();

    do_test(pre, false);
    do_test(pre, true);
}

ATF_TC(post);
ATF_TC_HEAD(post, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the POST macro");
}
ATF_TC_BODY(post, tc)
{
    require_ndebug();

    do_test(post, false);
    do_test(post, true);
}

ATF_TC(unreachable);
ATF_TC_HEAD(unreachable, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the UNREACHABLE macro");
}
ATF_TC_BODY(unreachable, tc)
{
    require_ndebug();

    do_test(unreachable, false);
    do_test(unreachable, true);
}

/* ---------------------------------------------------------------------
 * Main.
 * --------------------------------------------------------------------- */

ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, inv);
    ATF_TP_ADD_TC(tp, pre);
    ATF_TP_ADD_TC(tp, post);
    ATF_TP_ADD_TC(tp, unreachable);

    return atf_no_error();
}
