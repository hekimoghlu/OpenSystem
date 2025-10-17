/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <stdio.h>
#include <string.h>
#include <err.h>
#include <assert.h>
#include "windlocl.h"

#define MAX_LENGTH 10

struct testcase {
    uint32_t in[MAX_LENGTH];
    size_t ilen;
    uint32_t out[MAX_LENGTH];
    size_t olen;
};

static const struct testcase testcases[] = {
    { { 0x20 }, 1, { 0 }, 0 },
    { { 0x20, 0x61 }, 2, { 0x20, 0x61, 0x20}, 3 },
    { { 0x20, 0x61, 0x20 }, 3, { 0x20, 0x61, 0x20}, 3 },
    { { 0x20, 0x61, 0x20, 0x61 }, 4, { 0x20, 0x61, 0x20, 0x20, 0x61, 0x20}, 6 }
};

static const struct testcase testcases2[] = {
    { { 0x20 }, 1, { 0x20 }, 1 },
    { { 0x20, 0x41 }, 2, { 0x20, 0x61}, 2 }
};


int
main(void)
{
    uint32_t out[MAX_LENGTH];
    unsigned failures = 0;
    unsigned i;
    size_t olen;
    int ret;


    for (i = 0; i < sizeof(testcases)/sizeof(testcases[0]); ++i) {
	const struct testcase *t = &testcases[i];

	olen = sizeof(out)/sizeof(out[0]);
	assert(olen > t->olen);

	ret = _wind_ldap_case_exact_attribute(t->in, t->ilen, out, &olen);
	if (ret) {
	    printf("wlcea: %u: %d\n", i, ret);
	    ++failures;
	    continue;
	}
	if (olen != t->olen) {
	    printf("len wlcea: %u %u != %u\n", i,
		   (unsigned)olen, (unsigned)t->olen);
	    failures++;
	    continue;
	}
	if (memcmp(t->out, out, sizeof(out[0]) * olen) != 0) {
	    printf("memcmp wlcea: %u\n", i);
	    failures++;
	    continue;
	}
    }

    for (i = 0; i < sizeof(testcases2)/sizeof(testcases2[0]); ++i) {
	const struct testcase *t = &testcases2[i];

	olen = sizeof(out)/sizeof(out[0]);
	assert(olen > t->olen);

	ret = wind_stringprep(t->in, t->ilen, out, &olen,
			      WIND_PROFILE_LDAP_CASE);

	if (ret) {
	    printf("wsplc: %u: %d\n", i, ret);
	    ++failures;
	    continue;
	}

	if (olen != t->olen) {
	    printf("strlen wsplc: %u: %d\n", i, ret);
	    ++failures;
	    continue;
	}
	if (memcmp(t->out, out, sizeof(out[0]) * olen) != 0) {
	    printf("memcmp wsplc: %u\n", i);
	    failures++;
	    continue;
	}
    }

    return failures != 0;
}
