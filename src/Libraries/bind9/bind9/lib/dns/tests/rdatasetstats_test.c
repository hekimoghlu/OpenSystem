/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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

#include <isc/print.h>

#include <dns/stats.h>

#include "dnstest.h"

/*
 * Helper functions
 */
static void
set_typestats(dns_stats_t *stats, dns_rdatatype_t type,
	      isc_boolean_t stale)
{
	dns_rdatastatstype_t which;
	unsigned int attributes;

	attributes = 0;
	if (stale) attributes |= DNS_RDATASTATSTYPE_ATTR_STALE;
	which = DNS_RDATASTATSTYPE_VALUE(type, attributes);
	dns_rdatasetstats_increment(stats, which);

	attributes = DNS_RDATASTATSTYPE_ATTR_NXRRSET;
	if (stale) attributes |= DNS_RDATASTATSTYPE_ATTR_STALE;
	which = DNS_RDATASTATSTYPE_VALUE(type, attributes);
	dns_rdatasetstats_increment(stats, which);
}

static void
set_nxdomainstats(dns_stats_t *stats, isc_boolean_t stale) {
	dns_rdatastatstype_t which;
	unsigned int attributes;

	attributes = DNS_RDATASTATSTYPE_ATTR_NXDOMAIN;
	if (stale) attributes |= DNS_RDATASTATSTYPE_ATTR_STALE;
	which = DNS_RDATASTATSTYPE_VALUE(0, attributes);
	dns_rdatasetstats_increment(stats, which);
}

static void
checkit1(dns_rdatastatstype_t which, isc_uint64_t value, void *arg) {
	unsigned int attributes;
#if debug
	unsigned int type;
#endif

	UNUSED(which);
	UNUSED(arg);

	attributes = DNS_RDATASTATSTYPE_ATTR(which);
#if debug
	type = DNS_RDATASTATSTYPE_BASE(which);

	fprintf(stderr, "%s%s%s%s/%u, %u\n",
		attributes & DNS_RDATASTATSTYPE_ATTR_OTHERTYPE ? "O" : " ",
		attributes & DNS_RDATASTATSTYPE_ATTR_NXRRSET ? "!" : " ",
		attributes & DNS_RDATASTATSTYPE_ATTR_STALE ? "#" : " ",
		attributes & DNS_RDATASTATSTYPE_ATTR_NXDOMAIN ? "X" : " ",
		type, (unsigned)value);
#endif
	if ((attributes & DNS_RDATASTATSTYPE_ATTR_STALE) == 0)
		ATF_REQUIRE_EQ(value, 1);
	else
		ATF_REQUIRE_EQ(value, 0);
}

static void
checkit2(dns_rdatastatstype_t which, isc_uint64_t value, void *arg) {
	unsigned int attributes;
#if debug
	unsigned int type;
#endif

	UNUSED(which);
	UNUSED(arg);

	attributes = DNS_RDATASTATSTYPE_ATTR(which);
#if debug
	type = DNS_RDATASTATSTYPE_BASE(which);

	fprintf(stderr, "%s%s%s%s/%u, %u\n",
		attributes & DNS_RDATASTATSTYPE_ATTR_OTHERTYPE ? "O" : " ",
		attributes & DNS_RDATASTATSTYPE_ATTR_NXRRSET ? "!" : " ",
		attributes & DNS_RDATASTATSTYPE_ATTR_STALE ? "#" : " ",
		attributes & DNS_RDATASTATSTYPE_ATTR_NXDOMAIN ? "X" : " ",
		type, (unsigned)value);
#endif
	if ((attributes & DNS_RDATASTATSTYPE_ATTR_STALE) == 0)
		ATF_REQUIRE_EQ(value, 0);
	else
		ATF_REQUIRE_EQ(value, 1);
}
/*
 * Individual unit tests
 */

ATF_TC(rdatasetstats);
ATF_TC_HEAD(rdatasetstats, tc) {
	atf_tc_set_md_var(tc, "descr", "test that rdatasetstats counters are properly set");
}
ATF_TC_BODY(rdatasetstats, tc) {
	unsigned int i;
	dns_stats_t *stats = NULL;
	isc_result_t result;

	UNUSED(tc);

	result = dns_test_begin(NULL, ISC_TRUE);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	result = dns_rdatasetstats_create(mctx, &stats);
	ATF_REQUIRE_EQ(result, ISC_R_SUCCESS);

	/* First 256 types. */
	for (i = 0; i <= 255; i++)
		set_typestats(stats, (dns_rdatatype_t)i, ISC_FALSE);
	/* Specials */
	set_typestats(stats, dns_rdatatype_dlv, ISC_FALSE);
	set_typestats(stats, (dns_rdatatype_t)1000, ISC_FALSE);
	set_nxdomainstats(stats, ISC_FALSE);

	/*
	 * Check that all counters are set to appropriately.
	 */
	dns_rdatasetstats_dump(stats, checkit1, NULL, 1);

	/* First 256 types. */
	for (i = 0; i <= 255; i++)
		set_typestats(stats, (dns_rdatatype_t)i, ISC_TRUE);
	/* Specials */
	set_typestats(stats, dns_rdatatype_dlv, ISC_TRUE);
	set_typestats(stats, (dns_rdatatype_t)1000, ISC_TRUE);
	set_nxdomainstats(stats, ISC_TRUE);

	/*
	 * Check that all counters are set to appropriately.
	 */
	dns_rdatasetstats_dump(stats, checkit2, NULL, 1);

	dns_stats_detach(&stats);
	dns_test_end();
}

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
	ATF_TP_ADD_TC(tp, rdatasetstats);
	return (atf_no_error());
}

