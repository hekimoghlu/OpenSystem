/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

#include <isc/buffer.h>
#include <isc/socket.h>
#include <isc/task.h>
#include <isc/timer.h>

#include <dns/dispatch.h>
#include <dns/name.h>
#include <dns/view.h>

#include "dnstest.h"

dns_dispatchmgr_t *dispatchmgr = NULL;
dns_dispatchset_t *dset = NULL;

static isc_result_t
make_dispatchset(unsigned int ndisps) {
	isc_result_t result;
	isc_sockaddr_t any;
	unsigned int attrs;
	dns_dispatch_t *disp = NULL;

	result = dns_dispatchmgr_create(mctx, NULL, &dispatchmgr);
	if (result != ISC_R_SUCCESS)
		return (result);

	isc_sockaddr_any(&any);
	attrs = DNS_DISPATCHATTR_IPV4 | DNS_DISPATCHATTR_UDP;
	result = dns_dispatch_getudp(dispatchmgr, socketmgr, taskmgr,
				     &any, 512, 6, 1024, 17, 19, attrs,
				     attrs, &disp);
	if (result != ISC_R_SUCCESS)
		return (result);

	result = dns_dispatchset_create(mctx, socketmgr, taskmgr, disp,
					&dset, ndisps);
	dns_dispatch_detach(&disp);

	return (result);
}

static void
teardown(void) {
	if (dset != NULL)
		dns_dispatchset_destroy(&dset);
	if (dispatchmgr != NULL)
		dns_dispatchmgr_destroy(&dispatchmgr);
}

/*
 * Individual unit tests
 */
ATF_TC(dispatchset_create);
ATF_TC_HEAD(dispatchset_create, tc) {
	atf_tc_set_md_var(tc, "descr", "create dispatch set");
}
ATF_TC_BODY(dispatchset_create, tc) {
	isc_result_t result;

	UNUSED(tc);

	result = dns_test_begin(NULL, ISC_TRUE);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	result = make_dispatchset(1);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);
	teardown();

	result = make_dispatchset(10);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);
	teardown();

	dns_test_end();
}



ATF_TC(dispatchset_get);
ATF_TC_HEAD(dispatchset_get, tc) {
	atf_tc_set_md_var(tc, "descr", "test dispatch set round-robin");
}
ATF_TC_BODY(dispatchset_get, tc) {
	isc_result_t result;
	dns_dispatch_t *d1, *d2, *d3, *d4, *d5;

	UNUSED(tc);

	result = dns_test_begin(NULL, ISC_TRUE);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	result = make_dispatchset(1);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	d1 = dns_dispatchset_get(dset);
	d2 = dns_dispatchset_get(dset);
	d3 = dns_dispatchset_get(dset);
	d4 = dns_dispatchset_get(dset);
	d5 = dns_dispatchset_get(dset);

	ATF_CHECK_EQ(d1, d2);
	ATF_CHECK_EQ(d2, d3);
	ATF_CHECK_EQ(d3, d4);
	ATF_CHECK_EQ(d4, d5);

	teardown();

	result = make_dispatchset(4);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	d1 = dns_dispatchset_get(dset);
	d2 = dns_dispatchset_get(dset);
	d3 = dns_dispatchset_get(dset);
	d4 = dns_dispatchset_get(dset);
	d5 = dns_dispatchset_get(dset);

	ATF_CHECK_EQ(d1, d5);
	ATF_CHECK(d1 != d2);
	ATF_CHECK(d2 != d3);
	ATF_CHECK(d3 != d4);
	ATF_CHECK(d4 != d5);

	teardown();
	dns_test_end();
}


/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
	ATF_TP_ADD_TC(tp, dispatchset_create);
	ATF_TP_ADD_TC(tp, dispatchset_get);
	return (atf_no_error());
}

