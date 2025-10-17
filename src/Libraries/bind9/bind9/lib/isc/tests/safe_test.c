/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
/* $Id$ */

/* ! \file */

#include <config.h>

#include <atf-c.h>

#include <stdio.h>
#include <string.h>

#include <isc/safe.h>
#include <isc/util.h>

ATF_TC(isc_safe_memequal);
ATF_TC_HEAD(isc_safe_memequal, tc) {
	atf_tc_set_md_var(tc, "descr", "safe memequal()");
}
ATF_TC_BODY(isc_safe_memequal, tc) {
	UNUSED(tc);

	ATF_CHECK(isc_safe_memequal("test", "test", 4));
	ATF_CHECK(!isc_safe_memequal("test", "tesc", 4));
	ATF_CHECK(isc_safe_memequal("\x00\x00\x00\x00",
				    "\x00\x00\x00\x00", 4));
	ATF_CHECK(!isc_safe_memequal("\x00\x00\x00\x00",
				     "\x00\x00\x00\x01", 4));
	ATF_CHECK(!isc_safe_memequal("\x00\x00\x00\x02",
				     "\x00\x00\x00\x00", 4));
}

ATF_TC(isc_safe_memcompare);
ATF_TC_HEAD(isc_safe_memcompare, tc) {
	atf_tc_set_md_var(tc, "descr", "safe memcompare()");
}
ATF_TC_BODY(isc_safe_memcompare, tc) {
	UNUSED(tc);

	ATF_CHECK(isc_safe_memcompare("test", "test", 4) == 0);
	ATF_CHECK(isc_safe_memcompare("test", "tesc", 4) > 0);
	ATF_CHECK(isc_safe_memcompare("test", "tesy", 4) < 0);
	ATF_CHECK(isc_safe_memcompare("\x00\x00\x00\x00",
				      "\x00\x00\x00\x00", 4) == 0);
	ATF_CHECK(isc_safe_memcompare("\x00\x00\x00\x00",
				      "\x00\x00\x00\x01", 4) < 0);
	ATF_CHECK(isc_safe_memcompare("\x00\x00\x00\x02",
				      "\x00\x00\x00\x00", 4) > 0);
}

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
	ATF_TP_ADD_TC(tp, isc_safe_memequal);
	ATF_TP_ADD_TC(tp, isc_safe_memcompare);
	return (atf_no_error());
}
