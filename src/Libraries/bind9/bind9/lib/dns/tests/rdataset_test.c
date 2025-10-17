/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

#include <dns/rdataset.h>
#include <dns/rdatastruct.h>

#include "dnstest.h"


/*
 * Individual unit tests
 */

/* Successful load test */
ATF_TC(trimttl);
ATF_TC_HEAD(trimttl, tc) {
	atf_tc_set_md_var(tc, "descr", "dns_master_loadfile() loads a "
				       "valid master file and returns success");
}
ATF_TC_BODY(trimttl, tc) {
	isc_result_t result;
	dns_rdataset_t rdataset, sigrdataset;
	dns_rdata_rrsig_t rrsig;
	isc_stdtime_t ttltimenow, ttltimeexpire;

	ttltimenow = 10000000;
	ttltimeexpire = ttltimenow + 800;

	UNUSED(tc);

	dns_rdataset_init(&rdataset);
	dns_rdataset_init(&sigrdataset);

	result = dns_test_begin(NULL, ISC_FALSE);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	rdataset.ttl = 900;
	sigrdataset.ttl = 1000;
	rrsig.timeexpire = ttltimeexpire;
	rrsig.originalttl = 1000;

	dns_rdataset_trimttl(&rdataset, &sigrdataset, &rrsig, ttltimenow,
			     ISC_TRUE);
	ATF_REQUIRE_EQ(rdataset.ttl, 800);
	ATF_REQUIRE_EQ(sigrdataset.ttl, 800);

	rdataset.ttl = 900;
	sigrdataset.ttl = 1000;
	rrsig.timeexpire = ttltimenow - 200;
	rrsig.originalttl = 1000;

	dns_rdataset_trimttl(&rdataset, &sigrdataset, &rrsig, ttltimenow,
			     ISC_TRUE);
	ATF_REQUIRE_EQ(rdataset.ttl, 120);
	ATF_REQUIRE_EQ(sigrdataset.ttl, 120);

	rdataset.ttl = 900;
	sigrdataset.ttl = 1000;
	rrsig.timeexpire = ttltimenow - 200;
	rrsig.originalttl = 1000;

	dns_rdataset_trimttl(&rdataset, &sigrdataset, &rrsig, ttltimenow,
			     ISC_FALSE);
	ATF_REQUIRE_EQ(rdataset.ttl, 0);
	ATF_REQUIRE_EQ(sigrdataset.ttl, 0);

	sigrdataset.ttl = 900;
	rdataset.ttl = 1000;
	rrsig.timeexpire = ttltimeexpire;
	rrsig.originalttl = 1000;

	dns_rdataset_trimttl(&rdataset, &sigrdataset, &rrsig, ttltimenow,
			     ISC_TRUE);
	ATF_REQUIRE_EQ(rdataset.ttl, 800);
	ATF_REQUIRE_EQ(sigrdataset.ttl, 800);

	sigrdataset.ttl = 900;
	rdataset.ttl = 1000;
	rrsig.timeexpire = ttltimenow - 200;
	rrsig.originalttl = 1000;

	dns_rdataset_trimttl(&rdataset, &sigrdataset, &rrsig, ttltimenow,
			     ISC_TRUE);
	ATF_REQUIRE_EQ(rdataset.ttl, 120);
	ATF_REQUIRE_EQ(sigrdataset.ttl, 120);

	sigrdataset.ttl = 900;
	rdataset.ttl = 1000;
	rrsig.timeexpire = ttltimenow - 200;
	rrsig.originalttl = 1000;

	dns_rdataset_trimttl(&rdataset, &sigrdataset, &rrsig, ttltimenow,
			     ISC_FALSE);
	ATF_REQUIRE_EQ(rdataset.ttl, 0);
	ATF_REQUIRE_EQ(sigrdataset.ttl, 0);

	dns_test_end();
}

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
	ATF_TP_ADD_TC(tp, trimttl);

	return (atf_no_error());
}

