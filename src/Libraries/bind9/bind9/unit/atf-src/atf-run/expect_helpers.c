/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <atf-c.h>

ATF_TC_WITHOUT_HEAD(pass_and_pass);
ATF_TC_BODY(pass_and_pass, tc)
{
    atf_tc_expect_pass();
}

ATF_TC_WITHOUT_HEAD(pass_but_fail_requirement);
ATF_TC_BODY(pass_but_fail_requirement, tc)
{
    atf_tc_expect_pass();
    atf_tc_fail("Some reason");
}

ATF_TC_WITHOUT_HEAD(pass_but_fail_check);
ATF_TC_BODY(pass_but_fail_check, tc)
{
    atf_tc_expect_pass();
    atf_tc_fail_nonfatal("Some reason");
}

ATF_TC_WITHOUT_HEAD(fail_and_fail_requirement);
ATF_TC_BODY(fail_and_fail_requirement, tc)
{
    atf_tc_expect_fail("Fail %s", "reason");
    atf_tc_fail("The failure");
    atf_tc_expect_pass();
}

ATF_TC_WITHOUT_HEAD(fail_and_fail_check);
ATF_TC_BODY(fail_and_fail_check, tc)
{
    atf_tc_expect_fail("Fail first");
    atf_tc_fail_nonfatal("abc");
    atf_tc_expect_pass();

    atf_tc_expect_fail("And fail again");
    atf_tc_fail_nonfatal("def");
    atf_tc_expect_pass();
}

ATF_TC_WITHOUT_HEAD(fail_but_pass);
ATF_TC_BODY(fail_but_pass, tc)
{
    atf_tc_expect_fail("Fail first");
    atf_tc_fail_nonfatal("abc");
    atf_tc_expect_pass();

    atf_tc_expect_fail("Will not fail");
    atf_tc_expect_pass();

    atf_tc_expect_fail("And fail again");
    atf_tc_fail_nonfatal("def");
    atf_tc_expect_pass();
}

ATF_TC_WITHOUT_HEAD(exit_any_and_exit);
ATF_TC_BODY(exit_any_and_exit, tc)
{
    atf_tc_expect_exit(-1, "Call will exit");
    exit(EXIT_SUCCESS);
}

ATF_TC_WITHOUT_HEAD(exit_code_and_exit);
ATF_TC_BODY(exit_code_and_exit, tc)
{
    atf_tc_expect_exit(123, "Call will exit");
    exit(123);
}

ATF_TC_WITHOUT_HEAD(exit_but_pass);
ATF_TC_BODY(exit_but_pass, tc)
{
    atf_tc_expect_exit(-1, "Call won't exit");
}

ATF_TC_WITHOUT_HEAD(signal_any_and_signal);
ATF_TC_BODY(signal_any_and_signal, tc)
{
    atf_tc_expect_signal(-1, "Call will signal");
    kill(getpid(), SIGKILL);
}

ATF_TC_WITHOUT_HEAD(signal_no_and_signal);
ATF_TC_BODY(signal_no_and_signal, tc)
{
    atf_tc_expect_signal(SIGHUP, "Call will signal");
    kill(getpid(), SIGHUP);
}

ATF_TC_WITHOUT_HEAD(signal_but_pass);
ATF_TC_BODY(signal_but_pass, tc)
{
    atf_tc_expect_signal(-1, "Call won't signal");
}

ATF_TC_WITHOUT_HEAD(death_and_exit);
ATF_TC_BODY(death_and_exit, tc)
{
    atf_tc_expect_death("Exit case");
    exit(123);
}

ATF_TC_WITHOUT_HEAD(death_and_signal);
ATF_TC_BODY(death_and_signal, tc)
{
    atf_tc_expect_death("Signal case");
    kill(getpid(), SIGKILL);
}

ATF_TC_WITHOUT_HEAD(death_but_pass);
ATF_TC_BODY(death_but_pass, tc)
{
    atf_tc_expect_death("Call won't die");
}

ATF_TC(timeout_and_hang);
ATF_TC_HEAD(timeout_and_hang, tc)
{
    atf_tc_set_md_var(tc, "timeout", "1");
}
ATF_TC_BODY(timeout_and_hang, tc)
{
    atf_tc_expect_timeout("Will overrun");
    sleep(5);
}

ATF_TC(timeout_but_pass);
ATF_TC_HEAD(timeout_but_pass, tc)
{
    atf_tc_set_md_var(tc, "timeout", "1");
}
ATF_TC_BODY(timeout_but_pass, tc)
{
    atf_tc_expect_timeout("Will just exit");
}

ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, pass_and_pass);
    ATF_TP_ADD_TC(tp, pass_but_fail_requirement);
    ATF_TP_ADD_TC(tp, pass_but_fail_check);
    ATF_TP_ADD_TC(tp, fail_and_fail_requirement);
    ATF_TP_ADD_TC(tp, fail_and_fail_check);
    ATF_TP_ADD_TC(tp, fail_but_pass);
    ATF_TP_ADD_TC(tp, exit_any_and_exit);
    ATF_TP_ADD_TC(tp, exit_code_and_exit);
    ATF_TP_ADD_TC(tp, exit_but_pass);
    ATF_TP_ADD_TC(tp, signal_any_and_signal);
    ATF_TP_ADD_TC(tp, signal_no_and_signal);
    ATF_TP_ADD_TC(tp, signal_but_pass);
    ATF_TP_ADD_TC(tp, death_and_exit);
    ATF_TP_ADD_TC(tp, death_and_signal);
    ATF_TP_ADD_TC(tp, death_but_pass);
    ATF_TP_ADD_TC(tp, timeout_and_hang);
    ATF_TP_ADD_TC(tp, timeout_but_pass);

    return atf_no_error();
}
