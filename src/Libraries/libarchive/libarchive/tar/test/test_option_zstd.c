/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

DEFINE_TEST(test_option_zstd)
{
	char *p;
	int r;
	size_t s;

	/* Create a file. */
	assertMakeFile("f", 0644, "a");

	/* Archive it with lz4 compression. */
	r = systemf("%s -cf - --zstd f >archive.out 2>archive.err",
	    testprog);
	p = slurpfile(&s, "archive.err");
	p[s] = '\0';
	if (r != 0) {
		if (strstr(p, "Unsupported compression") != NULL) {
			skipping("This version of bsdtar was compiled "
			    "without zstd support");
			goto done;
		}
		/* POSIX permits different handling of the spawnp
		 * system call used to launch the subsidiary
		 * program: */
		/* Some systems fail immediately to spawn the new process. */
		if (strstr(p, "Can't launch") != NULL && !canZstd()) {
			skipping("This version of bsdtar uses an external zstd program "
			    "but no such program is available on this system.");
			goto done;
		}
		/* Some systems successfully spawn the new process,
		 * but fail to exec a program within that process.
		 * This results in failure at the first attempt to
		 * write. */
		if (strstr(p, "Can't write") != NULL && !canZstd()) {
			skipping("This version of bsdtar uses an external zstd program "
			    "but no such program is available on this system.");
			goto done;
		}
		/* On some systems the error won't be detected until closing
		   time, by a 127 exit error returned by waitpid. */
		if (strstr(p, "Error closing") != NULL && !canZstd()) {
			skipping("This version of bsdcpio uses an external zstd program "
			    "but no such program is available on this system.");
			return;
		}
		failure("--zstd option is broken: %s", p);
		assertEqualInt(r, 0);
		goto done;
	}
	free(p);
	/* Check that the archive file has an lz4 signature. */
	p = slurpfile(&s, "archive.out");
	assert(s > 2);
	assertEqualMem(p, "\x28\xb5\x2f\xfd", 4);

done:
	free(p);
}

