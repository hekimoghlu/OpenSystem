/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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

#include <atf-c.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../lwconfig.c"

static void
setup_test() {
	/*
	 * atf-run changes us to a /tmp directory, so tests
	 * that access test data files must first chdir to the proper
	 * location.
	 */
	ATF_CHECK(chdir(TESTS) != -1);
}

ATF_TC(parse_linklocal);
ATF_TC_HEAD(parse_linklocal, tc) {
	atf_tc_set_md_var(tc, "descr", "lwres_conf_parse link-local nameserver");
}
ATF_TC_BODY(parse_linklocal, tc) {
	lwres_result_t result;
	lwres_context_t *ctx = NULL;
	unsigned char addr[16] = { 0xfe, 0x80, 0x00, 0x00,
				   0x00, 0x00, 0x00, 0x00,
				   0x00, 0x00, 0x00, 0x00,
				   0x00, 0x00, 0x00, 0x01 };

	UNUSED(tc);

	setup_test();

	lwres_context_create(&ctx, NULL, NULL, NULL,
			     LWRES_CONTEXT_USEIPV4 | LWRES_CONTEXT_USEIPV6);
	ATF_CHECK_EQ(ctx->confdata.nsnext, 0);
	ATF_CHECK_EQ(ctx->confdata.nameservers[0].zone, 0);

	result = lwres_conf_parse(ctx, "testdata/link-local.conf");
	ATF_CHECK_EQ(result, LWRES_R_SUCCESS);
	ATF_CHECK_EQ(ctx->confdata.nsnext, 1);
	ATF_CHECK_EQ(ctx->confdata.nameservers[0].zone, 1);
	ATF_CHECK_EQ(memcmp(ctx->confdata.nameservers[0].address, addr, 16), 0);
	lwres_context_destroy(&ctx);
}

/*
 * Main
 */
ATF_TP_ADD_TCS(tp) {
	ATF_TP_ADD_TC(tp, parse_linklocal);
	return (atf_no_error());
}
