/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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

DEFINE_TEST(test_option_d)
{
	int r;

	/*
	 * Create a file in a directory.
	 */
	assertMakeDir("dir", 0755);
	assertMakeFile("dir/file", 0644, NULL);

	/* Create an archive. */
	r = systemf("echo dir/file | %s -o > archive.cpio 2>archive.err", testprog);
	assertEqualInt(r, 0);
	assertTextFileContents("1 block\n", "archive.err");
	assertFileSize("archive.cpio", 512);

	/* Dearchive without -d, this should fail. */
	assertMakeDir("without-d", 0755);
	assertChdir("without-d");
	r = systemf("%s -i < ../archive.cpio >out 2>err", testprog);
	assertEqualInt(r, 0);
	assertEmptyFile("out");
	/* And the file should not be restored. */
	assertFileNotExists("dir/file");

	/* Dearchive with -d, this should succeed. */
	assertChdir("..");
	assertMakeDir("with-d", 0755);
	assertChdir("with-d");
	r = systemf("%s -id < ../archive.cpio >out 2>err", testprog);
	assertEqualInt(r, 0);
	assertEmptyFile("out");
	assertTextFileContents("1 block\n", "err");
	/* And the file should be restored. */
	assertFileExists("dir/file");
}
