/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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

/*! \file */

#include <config.h>

#include <atf-c.h>

#include <unistd.h>
#include <time.h>

#include <isc/parseint.h>

#include "isctest.h"

/*
 * Individual unit tests
 */

/* Test for 32 bit overflow on 64 bit machines in isc_parse_uint32 */
ATF_TC(parse_overflow);
ATF_TC_HEAD(parse_overflow, tc) {
	atf_tc_set_md_var(tc, "descr", "Check for 32 bit overflow");
}
ATF_TC_BODY(parse_overflow, tc) {
	isc_result_t result;
	isc_uint32_t output;
	UNUSED(tc);

	result = isc_test_begin(NULL, ISC_TRUE);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	result = isc_parse_uint32(&output, "1234567890", 10);
	ATF_CHECK_EQ(ISC_R_SUCCESS, result);
	ATF_CHECK_EQ(1234567890, output);

	result = isc_parse_uint32(&output, "123456789012345", 10);
	ATF_CHECK_EQ(ISC_R_RANGE, result);

	result = isc_parse_uint32(&output, "12345678901234567890", 10);
	ATF_CHECK_EQ(ISC_R_RANGE, result);

	isc_test_end();
}

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
	ATF_TP_ADD_TC(tp, parse_overflow);

	return (atf_no_error());
}

