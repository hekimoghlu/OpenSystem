/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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

#include <dns/db.h>
#include <dns/nsec3.h>

#include "dnstest.h"

#if defined(OPENSSL) || defined(PKCS11CRYPTO)
/*
 * Helper functions
 */

static void
iteration_test(const char *file, unsigned int expected) {
	isc_result_t result;
	dns_db_t *db = NULL;
	unsigned int iterations;

	result = dns_test_loaddb(&db, dns_dbtype_zone, "test", file);
	ATF_CHECK_EQ_MSG(result, ISC_R_SUCCESS, "%s", file);

	result = dns_nsec3_maxiterations(db, NULL, mctx, &iterations);
	ATF_CHECK_EQ_MSG(result, ISC_R_SUCCESS, "%s", file);

	ATF_CHECK_EQ_MSG(iterations, expected, "%s", file);

	dns_db_detach(&db);
}

/*
 * Individual unit tests
 */

ATF_TC(max_iterations);
ATF_TC_HEAD(max_iterations, tc) {
	atf_tc_set_md_var(tc, "descr", "check that appropriate max iterations "
			  " is returned for different key size mixes");
}
ATF_TC_BODY(max_iterations, tc) {
	isc_result_t result;

	UNUSED(tc);

	result = dns_test_begin(NULL, ISC_FALSE);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	iteration_test("testdata/nsec3/1024.db", 150);
	iteration_test("testdata/nsec3/2048.db", 500);
	iteration_test("testdata/nsec3/4096.db", 2500);
	iteration_test("testdata/nsec3/min-1024.db", 150);
	iteration_test("testdata/nsec3/min-2048.db", 500);

	dns_test_end();
}
#else
ATF_TC(untested);
ATF_TC_HEAD(untested, tc) {
	atf_tc_set_md_var(tc, "descr", "skipping nsec3 test");
}
ATF_TC_BODY(untested, tc) {
	UNUSED(tc);
	atf_tc_skip("DNSSEC not available");
}
#endif

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
#if defined(OPENSSL) || defined(PKCS11CRYPTO)
	ATF_TP_ADD_TC(tp, max_iterations);
#else
	ATF_TP_ADD_TC(tp, untested);
#endif

	return (atf_no_error());
}

