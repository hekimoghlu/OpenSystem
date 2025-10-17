/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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
#include <sys/param.h>
#include <sys/types.h>

#include <limits.h>
#include <stdio.h>
#include <unistd.h>

#include <atf-c.h>

#include "test_helpers.h"
#include "user.h"

/* ---------------------------------------------------------------------
 * Test cases for the free functions.
 * --------------------------------------------------------------------- */

ATF_TC(euid);
ATF_TC_HEAD(euid, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_user_euid function");
}
ATF_TC_BODY(euid, tc)
{
    ATF_REQUIRE_EQ(atf_user_euid(), geteuid());
}

ATF_TC(is_member_of_group);
ATF_TC_HEAD(is_member_of_group, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_user_is_member_of_group "
                      "function");
}
ATF_TC_BODY(is_member_of_group, tc)
{
    gid_t gids[NGROUPS_MAX];
    gid_t g, maxgid;
    int ngids;
    const gid_t maxgid_limit = 1 << 16;

    {
        int i;

        ngids = getgroups(NGROUPS_MAX, gids);
        if (ngids == -1)
            atf_tc_fail("Call to getgroups failed");
        maxgid = 0;
        for (i = 0; i < ngids; i++) {
            printf("User group %d is %u\n", i, gids[i]);
            if (maxgid < gids[i])
                maxgid = gids[i];
        }
        printf("User belongs to %d groups\n", ngids);
        printf("Last GID is %u\n", maxgid);
    }

    if (maxgid > maxgid_limit) {
        printf("Test truncated from %u groups to %u to keep the run time "
               "reasonable enough\n", maxgid, maxgid_limit);
        maxgid = maxgid_limit;
    }

    for (g = 0; g < maxgid; g++) {
        bool found = false;
        int i;

        for (i = 0; !found && i < ngids; i++) {
            if (gids[i] == g)
                found = true;
        }

        if (found) {
            printf("Checking if user belongs to group %d\n", g);
            ATF_REQUIRE(atf_user_is_member_of_group(g));
        } else {
            printf("Checking if user does not belong to group %d\n", g);
            ATF_REQUIRE(!atf_user_is_member_of_group(g));
        }
    }
}

ATF_TC(is_root);
ATF_TC_HEAD(is_root, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_user_is_root function");
}
ATF_TC_BODY(is_root, tc)
{
    if (geteuid() == 0)
        ATF_REQUIRE(atf_user_is_root());
    else
        ATF_REQUIRE(!atf_user_is_root());
}

ATF_TC(is_unprivileged);
ATF_TC_HEAD(is_unprivileged, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests the atf_user_is_unprivileged "
                      "function");
}
ATF_TC_BODY(is_unprivileged, tc)
{
    if (geteuid() != 0)
        ATF_REQUIRE(atf_user_is_unprivileged());
    else
        ATF_REQUIRE(!atf_user_is_unprivileged());
}

/* ---------------------------------------------------------------------
 * Main.
 * --------------------------------------------------------------------- */

ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, euid);
    ATF_TP_ADD_TC(tp, is_member_of_group);
    ATF_TP_ADD_TC(tp, is_root);
    ATF_TP_ADD_TC(tp, is_unprivileged);

    return atf_no_error();
}
