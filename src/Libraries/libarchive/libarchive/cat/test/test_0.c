/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

/*
 * This first test does basic sanity checks on the environment.  For
 * most of these, we just exit on failure.
 */
#if !defined(_WIN32) || defined(__CYGWIN__)
#define DEV_NULL "/dev/null"
#else
#define DEV_NULL "NUL"
#endif

DEFINE_TEST(test_0)
{
	struct stat st;

	failure("File %s does not exist?!", testprog);
	if (!assertEqualInt(0, stat(testprogfile, &st))) {
		fprintf(stderr,
		    "\nFile %s does not exist; aborting test.\n\n",
		    testprog);
		exit(1);
	}

	failure("%s is not executable?!", testprog);
	if (!assert((st.st_mode & 0111) != 0)) {
		fprintf(stderr,
		    "\nFile %s not executable; aborting test.\n\n",
		    testprog);
		exit(1);
	}

	/*
	 * Try to successfully run the program; this requires that
	 * we know some option that will succeed.
	 */
	if (0 != systemf("%s --version >" DEV_NULL, testprog)) {
		failure("Unable to successfully run: %s --version\n", testprog);
		assert(0);
	}

	/* TODO: Ensure that our reference files are available. */
}
