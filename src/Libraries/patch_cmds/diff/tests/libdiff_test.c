/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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

#include <TargetConditionals.h>
#if !TARGET_OS_OSX
int main(void) { }
#else /* TARGET_OS_OSX */
/*-
 * Copyright (c) 2024 Klara, Inc.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <sys/types.h>
#include <sys/mman.h>

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <arraylist.h>
#include <diff_main.h>

#include <atf-c.h>

ATF_TC_WITH_CLEANUP(diff_atomize_truncated);
ATF_TC_HEAD(diff_atomize_truncated, tc)
{
	atf_tc_set_md_var(tc, "descr", "Verify that the atomizer "
	    "does not crash when an input file is truncated");
}
ATF_TC_BODY(diff_atomize_truncated, tc)
{
	char line[128];
	struct diff_config cfg = { .atomize_func = diff_atomize_text_by_line };
	struct diff_data d = { };
	const char *fn = atf_tc_get_ident(tc);
	FILE *f;
	unsigned char *p;
	size_t size = 65536;

	ATF_REQUIRE((f = fopen(fn, "w+")) != NULL);
	line[sizeof(line) - 1] = '\n';
	for (unsigned int i = 0; i <= size / sizeof(line); i++) {
		memset(line, 'a' + i % 26, sizeof(line) - 1);
		ATF_REQUIRE(fwrite(line, sizeof(line), 1, f) == 1);
	}
	ATF_REQUIRE(fsync(fileno(f)) == 0);
	rewind(f);
	ATF_REQUIRE(truncate(fn, size / 2) == 0);
	ATF_REQUIRE((p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fileno(f), 0)) != MAP_FAILED);
	ATF_REQUIRE(diff_atomize_file(&d, &cfg, f, p, size, 0) == 0);
	ATF_REQUIRE((size_t)d.len <= size / 2);
	ATF_REQUIRE((size_t)d.len >= size / 2 - sizeof(line));
	ATF_REQUIRE(d.atomizer_flags & DIFF_ATOMIZER_FILE_TRUNCATED);
}
ATF_TC_CLEANUP(diff_atomize_truncated, tc)
{
	unlink(atf_tc_get_ident(tc));
}

ATF_TP_ADD_TCS(tp)
{
	ATF_TP_ADD_TC(tp, diff_atomize_truncated);
	return atf_no_error();
}
#endif /* TARGET_OS_OSX */
