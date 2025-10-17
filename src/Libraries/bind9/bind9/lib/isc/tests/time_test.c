/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
#include <config.h>
#include <stdlib.h>

#include <atf-c.h>

#include <isc/time.h>
#include <isc/result.h>

ATF_TC(isc_time_parsehttptimestamp);
ATF_TC_HEAD(isc_time_parsehttptimestamp, tc) {
	atf_tc_set_md_var(tc, "descr", "parse http time stamp");
}
ATF_TC_BODY(isc_time_parsehttptimestamp, tc) {
	isc_result_t result;
	isc_time_t t, x;
	char buf[ISC_FORMATHTTPTIMESTAMP_SIZE];

	setenv("TZ", "PST8PDT", 1);
	result = isc_time_now(&t);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	isc_time_formathttptimestamp(&t, buf, sizeof(buf));
	result = isc_time_parsehttptimestamp(buf, &x);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);
	ATF_REQUIRE_EQ(isc_time_seconds(&t), isc_time_seconds(&x));
}

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
	ATF_TP_ADD_TC(tp, isc_time_parsehttptimestamp);
	return (atf_no_error());
}

