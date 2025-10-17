/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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

#include <unistd.h>

#include <isc/util.h>
#include <isc/string.h>

#include <pk11/site.h>

#include <dns/name.h>
#include <dst/result.h>

#include "../dst_internal.h"

#include "dnstest.h"

#if defined(OPENSSL) && !defined(PK11_DH_DISABLE)

ATF_TC(isc_dh_computesecret);
ATF_TC_HEAD(isc_dh_computesecret, tc) {
	atf_tc_set_md_var(tc, "descr", "OpenSSL DH_compute_key() failure");
}
ATF_TC_BODY(isc_dh_computesecret, tc) {
	dst_key_t *key = NULL;
	isc_buffer_t buf;
	unsigned char array[1024];
	isc_result_t ret;
	dns_fixedname_t fname;
	dns_name_t *name;

	UNUSED(tc);

	ret = dns_test_begin(NULL, ISC_FALSE);
	ATF_REQUIRE_EQ(ret, ISC_R_SUCCESS);

	dns_fixedname_init(&fname);
	name = dns_fixedname_name(&fname);
	isc_buffer_constinit(&buf, "dh.", 3);
	isc_buffer_add(&buf, 3);
	ret = dns_name_fromtext(name, &buf, NULL, 0, NULL);
	ATF_REQUIRE_EQ(ret, ISC_R_SUCCESS);

	ret = dst_key_fromfile(name, 18602, DST_ALG_DH,
			       DST_TYPE_PUBLIC | DST_TYPE_KEY,
			       "./", mctx, &key);
	ATF_REQUIRE_EQ(ret, ISC_R_SUCCESS);

	isc_buffer_init(&buf, array, sizeof(array));
	ret = dst_key_computesecret(key, key, &buf);
	ATF_REQUIRE_EQ(ret, DST_R_NOTPRIVATEKEY);
	ret = key->func->computesecret(key, key, &buf);
	ATF_REQUIRE_EQ(ret, DST_R_COMPUTESECRETFAILURE);

	dst_key_free(&key);
	dns_test_end();
}
#else
ATF_TC(untested);
ATF_TC_HEAD(untested, tc) {
	atf_tc_set_md_var(tc, "descr", "skipping OpenSSL DH test");
}
ATF_TC_BODY(untested, tc) {
	UNUSED(tc);
	atf_tc_skip("OpenSSL DH not compiled in");
}
#endif
/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
#if defined(OPENSSL) && !defined(PK11_DH_DISABLE)
	ATF_TP_ADD_TC(tp, isc_dh_computesecret);
#else
	ATF_TP_ADD_TC(tp, untested);
#endif
	return (atf_no_error());
}
