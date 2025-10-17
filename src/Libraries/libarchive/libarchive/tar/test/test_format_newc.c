/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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

DEFINE_TEST(test_format_newc)
{

	assertMakeFile("file1", 0644, "file1");
	assertMakeFile("file2", 0644, "file2");
	assertMakeHardlink("file3", "file1");

	/* Test 1: Create an archive file with a newc format. */
	assertEqualInt(0,
	    systemf("%s -cf test1.cpio --format newc file1 file2 file3",
	    testprog));
	assertMakeDir("test1", 0755);
	assertChdir("test1");
	assertEqualInt(0,
	    systemf("%s -xf ../test1.cpio >test.out 2>test.err", testprog));
	assertFileContents("file1", 5, "file1");
	assertFileContents("file2", 5, "file2");
	assertFileContents("file1", 5, "file3");
	assertEmptyFile("test.out");
	assertEmptyFile("test.err");
	assertChdir("..");

	/* Test 2: Exclude one of hardlinked files. */
	assertEqualInt(0,
	    systemf("%s -cf test2.cpio --format newc file1 file2",
	    testprog));
	assertMakeDir("test2", 0755);
	assertChdir("test2");
	assertEqualInt(0,
	    systemf("%s -xf ../test2.cpio >test.out 2>test.err", testprog));
	assertFileContents("file1", 5, "file1");
	assertFileContents("file2", 5, "file2");
	assertFileNotExists("file3");
	assertEmptyFile("test.out");
	assertEmptyFile("test.err");
	assertChdir("..");
}
