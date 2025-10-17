/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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

#include <isc/counter.h>
#include <isc/result.h>

#include "isctest.h"

ATF_TC(isc_counter);
ATF_TC_HEAD(isc_counter, tc) {
	atf_tc_set_md_var(tc, "descr", "isc counter object");
}
ATF_TC_BODY(isc_counter, tc) {
	isc_result_t result;
	isc_counter_t *counter = NULL;
	int i;

	result = isc_test_begin(NULL, ISC_TRUE);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	result = isc_counter_create(mctx, 0, &counter);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	for (i = 0; i < 10; i++) {
		result = isc_counter_increment(counter);
		ATF_CHECK_EQ(result, ISC_R_SUCCESS);
	}

	ATF_CHECK_EQ(isc_counter_used(counter), 10);

	isc_counter_setlimit(counter, 15);
	for (i = 0; i < 10; i++) {
		result = isc_counter_increment(counter);
		if (result != ISC_R_SUCCESS)
			break;
	}

	ATF_CHECK_EQ(isc_counter_used(counter), 15);

	isc_counter_detach(&counter);
	isc_test_end();
}

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
	ATF_TP_ADD_TC(tp, isc_counter);
	return (atf_no_error());
}

