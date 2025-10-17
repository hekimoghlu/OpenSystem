/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
#include <atf-c.h>

#include "atf-c/error.h"

ATF_TC(pass);
ATF_TC_HEAD(pass, tc)
{
    atf_tc_set_md_var(tc, "descr", "An empty test case that always passes");
}
ATF_TC_BODY(pass, tc)
{
    atf_tc_pass();
}

ATF_TC(fail);
ATF_TC_HEAD(fail, tc)
{
    atf_tc_set_md_var(tc, "descr", "An empty test case that always fails");
}
ATF_TC_BODY(fail, tc)
{
    atf_tc_fail("On purpose");
}

ATF_TC(skip);
ATF_TC_HEAD(skip, tc)
{
    atf_tc_set_md_var(tc, "descr", "An empty test case that is always "
                      "skipped");
}
ATF_TC_BODY(skip, tc)
{
    atf_tc_skip("By design");
}

ATF_TC(default);
ATF_TC_HEAD(default, tc)
{
    atf_tc_set_md_var(tc, "descr", "A test case that passes without "
                      "explicitly stating it");
}
ATF_TC_BODY(default, tc)
{
}

ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, pass);
    ATF_TP_ADD_TC(tp, fail);
    ATF_TP_ADD_TC(tp, skip);
    ATF_TP_ADD_TC(tp, default);

    return atf_no_error();
}
