/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include "roken.h"
#include "parse_time.h"
#include "test-mem.h"
#include "err.h"

static struct testcase {
    size_t size;
    int    val;
    char  *str;
} tests[] = {
    { 8, 1,		"1 second" },
    { 17, 61,		"1 minute 1 second" },
    { 18, 62,		"1 minute 2 seconds" },
    { 8, 60,		"1 minute" },
    { 6, 3600,	 	"1 hour" },
    { 15, 3601,	 	"1 hour 1 second" },
    { 16, 3602,	 	"1 hour 2 seconds" }
};

int
main(int argc, char **argv)
{
    size_t sz;
    size_t buf_sz;
    int i, j;

    for (i = 0; i < sizeof(tests)/sizeof(tests[0]); ++i) {
	char *buf;

	sz = unparse_time(tests[i].val, NULL, 0);
	if  (sz != tests[i].size)
	    errx(1, "sz (%lu) != tests[%d].size (%lu)",
		 (unsigned long)sz, i, (unsigned long)tests[i].size);

	for (buf_sz = 0; buf_sz < tests[i].size + 2; buf_sz++) {

	    buf = rk_test_mem_alloc(RK_TM_OVERRUN, "overrun",
				    NULL, buf_sz);
	    sz = unparse_time(tests[i].val, buf, buf_sz);
	    if (sz != tests[i].size)
		errx(1, "sz (%lu) != tests[%d].size (%lu) with in size %lu",
		     (unsigned long)sz, i,
		     (unsigned long)tests[i].size,
		     (unsigned long)buf_sz);
	    if (buf_sz > 0 && memcmp(buf, tests[i].str, buf_sz - 1) != 0)
		errx(1, "test %i wrong result %s vs %s", i, buf, tests[i].str);
	    if (buf_sz > 0 && buf[buf_sz - 1] != '\0')
		errx(1, "test %i not zero terminated", i);
	    rk_test_mem_free("overrun");

	    buf = rk_test_mem_alloc(RK_TM_UNDERRUN, "underrun",
				    NULL, tests[i].size);
	    sz = unparse_time(tests[i].val, buf, min(buf_sz, tests[i].size));
	    if (sz != tests[i].size)
		errx(1, "sz (%lu) != tests[%d].size (%lu) with insize %lu",
		     (unsigned long)sz, i,
		     (unsigned long)tests[i].size,
		     (unsigned long)buf_sz);
	    if (buf_sz > 0 && strncmp(buf, tests[i].str, min(buf_sz, tests[i].size) - 1) != 0)
		errx(1, "test %i wrong result %s vs %s", i, buf, tests[i].str);
	    if (buf_sz > 0 && buf[min(buf_sz, tests[i].size) - 1] != '\0')
		errx(1, "test %i not zero terminated", i);
	    rk_test_mem_free("underrun");
	}

	buf = rk_test_mem_alloc(RK_TM_OVERRUN, "overrun",
				tests[i].str, tests[i].size + 1);
	j = parse_time(buf, "s");
	if (j != tests[i].val)
	    errx(1, "parse_time failed for test %d", i);
	rk_test_mem_free("overrun");

	buf = rk_test_mem_alloc(RK_TM_UNDERRUN, "underrun",
				tests[i].str, tests[i].size + 1);
	j = parse_time(buf, "s");
	if (j != tests[i].val)
	    errx(1, "parse_time failed for test %d", i);
	rk_test_mem_free("underrun");

    }
    return 0;
}
