/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
/* ! \file */

#include <config.h>

#include <atf-c.h>

#include <stdio.h>
#include <string.h>

#include <isc/netaddr.h>

ATF_TC(isc_netaddr_isnetzero);
ATF_TC_HEAD(isc_netaddr_isnetzero, tc) {
	atf_tc_set_md_var(tc, "descr", "test isc_netaddr_isnetzero");
}
ATF_TC_BODY(isc_netaddr_isnetzero, tc) {
	unsigned int i;
	struct in_addr ina;
	struct {
		const char *address;
		isc_boolean_t expect;
	} tests[] = {
		{ "0.0.0.0", ISC_TRUE },
		{ "0.0.0.1", ISC_TRUE },
		{ "0.0.1.2", ISC_TRUE },
		{ "0.1.2.3", ISC_TRUE },
		{ "10.0.0.0", ISC_FALSE },
		{ "10.9.0.0", ISC_FALSE },
		{ "10.9.8.0", ISC_FALSE },
		{ "10.9.8.7", ISC_FALSE },
		{ "127.0.0.0", ISC_FALSE },
		{ "127.0.0.1", ISC_FALSE }
	};
	isc_boolean_t result;
	isc_netaddr_t netaddr;

	for (i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) {
		ina.s_addr = inet_addr(tests[i].address);
		isc_netaddr_fromin(&netaddr, &ina);
		result = isc_netaddr_isnetzero(&netaddr);
		ATF_CHECK_EQ_MSG(result, tests[i].expect,
				 "%s", tests[i].address);
	}
}

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {

	ATF_TP_ADD_TC(tp, isc_netaddr_isnetzero);

	return (atf_no_error());
}
