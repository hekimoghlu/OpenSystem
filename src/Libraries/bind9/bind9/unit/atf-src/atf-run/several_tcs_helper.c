/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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

ATF_TC(first);
ATF_TC_HEAD(first, tc)
{
    atf_tc_set_md_var(tc, "descr", "Description 1");
}
ATF_TC_BODY(first, tc)
{
}

ATF_TC_WITH_CLEANUP(second);
ATF_TC_HEAD(second, tc)
{
    atf_tc_set_md_var(tc, "descr", "Description 2");
    atf_tc_set_md_var(tc, "timeout", "500");
    atf_tc_set_md_var(tc, "X-property", "Custom property");
}
ATF_TC_BODY(second, tc)
{
}
ATF_TC_CLEANUP(second, tc)
{
}

ATF_TC_WITHOUT_HEAD(third);
ATF_TC_BODY(third, tc)
{
}

ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, first);
    ATF_TP_ADD_TC(tp, second);
    ATF_TP_ADD_TC(tp, third);

    return atf_no_error();
}
