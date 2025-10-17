/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#include "test.h"
#if defined(HAVE_UTIME_H)
#include <utime.h>
#elif defined(HAVE_SYS_UTIME_H)
#include <sys/utime.h>
#endif

DEFINE_TEST(test_option_u)
{
	struct utimbuf times;
	char *p;
	size_t s;
	int r;

	/* Create a file. */
	assertMakeFile("f", 0644, "a");

	/* Copy the file to the "copy" dir. */
	r = systemf("echo f| %s -pd copy >copy.out 2>copy.err",
	    testprog);
	assertEqualInt(r, 0);

	/* Check that the file contains only a single "a" */
	p = slurpfile(&s, "copy/f");
	assertEqualInt(s, 1);
	assertEqualMem(p, "a", 1);
	free(p);

	/* Recreate the file with a single "b" */
	assertMakeFile("f", 0644, "b");

	/* Set the mtime to the distant past. */
	memset(&times, 0, sizeof(times));
	times.actime = 1;
	times.modtime = 3;
	assertEqualInt(0, utime("f", &times));

	/* Copy the file to the "copy" dir. */
	r = systemf("echo f| %s -pd copy >copy.out 2>copy.err",
	    testprog);
	assertEqualInt(r, 0);

	/* Verify that the file hasn't changed (it wasn't overwritten) */
	p = slurpfile(&s, "copy/f");
	assertEqualInt(s, 1);
	assertEqualMem(p, "a", 1);
	free(p);

	/* Copy the file to the "copy" dir with -u (force) */
	r = systemf("echo f| %s -pud copy >copy.out 2>copy.err",
	    testprog);
	assertEqualInt(r, 0);

	/* Verify that the file has changed (it was overwritten) */
	p = slurpfile(&s, "copy/f");
	assertEqualInt(s, 1);
	assertEqualMem(p, "b", 1);
	free(p);
}
