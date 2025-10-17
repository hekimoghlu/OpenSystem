/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#include <sys/cdefs.h>

#include <sys/param.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <darwintest.h>
#include <darwintest_utils.h>

/*
 * Derived from Russ Cox' pathological case test program used for the
 * https://research.swtch.com/glob article.
 */
T_DECL(glob_pathological_test, "Russ Cox's pathological test program")
{
	struct timespec t, t2;
	glob_t g;
	const char *longname = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
	char pattern[1000], *p;
	double dt;
	unsigned i, j, k, mul = 10;
	int fd, rc;

	T_SETUPBEGIN;
	rc = chdir(dt_tmpdir());
	T_ASSERT_POSIX_ZERO(rc, NULL);
	fd = open(longname, O_CREAT | O_RDWR, 0666);
	T_ASSERT_POSIX_SUCCESS(fd, NULL);
	T_SETUPEND;

	/*
	 * Test up to 100 a* groups.  Exponential implementations typically go
	 * bang at i=7 or 8.
	 */
	for (i = 0; i < 100; i++) {
		/*
		 * Create a*...b pattern with i 'a*' groups.
		 */
		p = pattern;
		for (k = 0; k < i; k++) {
			*p++ = 'a';
			*p++ = '*';
		}
		*p++ = 'b';
		*p = '\0';

		clock_gettime(CLOCK_MONOTONIC_RAW, &t);
		for (j = 0; j < mul; j++) {
			memset(&g, 0, sizeof g);
			rc = glob(pattern, 0, 0, &g);
			if (rc == GLOB_NOSPACE || rc == GLOB_ABORTED) {
				T_ASSERT_EQ(rc, GLOB_NOMATCH,
				    "an unexpected error occurred: "
				    "rc=%d errno=%d", rc, errno);
				/* NORETURN */
			}

			if (rc != GLOB_NOMATCH) {
			    T_FAIL("A bogus match occurred: '%s' ~ '%s'", pattern, g.gl_pathv ? g.gl_pathv[0] : "(NULL)");
			}
			globfree(&g);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);

		t2.tv_sec -= t.tv_sec;
		t2.tv_nsec -= t.tv_nsec;
		dt = t2.tv_sec + (double)t2.tv_nsec/1e9;
		dt /= mul;

		T_ASSERT_LE(dt, 1.0, "glob(3) completes in reasonable time (%d): %.9f sec/match", i,
		    dt);

		if (dt >= 0.0001)
			mul = 1;
	}
}
