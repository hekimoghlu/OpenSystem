/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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

DEFINE_TEST(test_option_C_upper)
{
	int r;

	/*
	 * Create a file on disk.
	 */
	assertMakeFile("file", 0644, NULL);

	/* Create an archive without -C; this should be 512 bytes. */
	r = systemf("echo file | %s -o > small.cpio 2>small.err", testprog);
	assertEqualInt(r, 0);
	assertTextFileContents("1 block\n", "small.err");
	assertFileSize("small.cpio", 512);

	/* Create an archive with -C 513; this should be 513 bytes. */
	r = systemf("echo file | %s -o -C 513 > 513.cpio 2>513.err",
		    testprog);
	assertEqualInt(r, 0);
	assertTextFileContents("1 block\n", "513.err");
	assertFileSize("513.cpio", 513);

	/* Create an archive with -C 12345; this should be 12345 bytes. */
	r = systemf("echo file | %s -o -C12345 > 12345.cpio 2>12345.err",
		    testprog);
	assertEqualInt(r, 0);
	assertTextFileContents("1 block\n", "12345.err");
	assertFileSize("12345.cpio", 12345);

	/* Create an archive with invalid -C request */
	assert(0 != systemf("echo file | %s -o -C > bad.cpio 2>bad.err",
			    testprog));
	assertEmptyFile("bad.cpio");
}
