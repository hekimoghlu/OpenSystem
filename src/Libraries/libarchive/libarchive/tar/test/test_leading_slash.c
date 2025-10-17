/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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

DEFINE_TEST(test_leading_slash)
{
	const char *reffile = "test_leading_slash.tar";
	char *errfile;
	size_t errfile_size;
	const char *expected_errmsg = "Removing leading '/' from member names";

	extract_reference_file(reffile);
	assertEqualInt(0, systemf("%s -xf %s >test.out 2>test.err", testprog, reffile));
	assertFileExists("foo/file");
	assertTextFileContents("foo\x0a", "foo/file");
	assertTextFileContents("foo\x0a", "foo/hardlink");
	assertIsHardlink("foo/file", "foo/hardlink");
	assertEmptyFile("test.out");

	/* Verify the error output contains the expected text somewhere in it */
	if (assertFileExists("test.err")) {
		errfile = slurpfile(&errfile_size, "test.err");
		assert(strstr(errfile, expected_errmsg) != NULL);
		free(errfile);
	}
}

