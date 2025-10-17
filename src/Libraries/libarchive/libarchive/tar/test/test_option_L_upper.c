/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

DEFINE_TEST(test_option_L_upper)
{

	if (!canSymlink()) {
		skipping("Can't test symlinks on this filesystem");
		return;
	}

	/*
	 * Create a sample archive.
	 */
	assertMakeDir("in", 0755);
	assertChdir("in");
	assertMakeDir("d1", 0755);
	assertMakeSymlink("ld1", "d1", 1);
	assertMakeFile("d1/file1", 0644, "d1/file1");
	assertMakeFile("d1/file2", 0644, "d1/file2");
	assertMakeSymlink("d1/link1", "file1", 0);
	assertMakeSymlink("d1/linkX", "fileX", 0);
	assertMakeSymlink("link2", "d1/file2", 0);
	assertMakeSymlink("linkY", "d1/fileY", 0);
	assertChdir("..");

	/* Test 1: Without -L */
	assertMakeDir("test1", 0755);
	assertEqualInt(0,
	    systemf("%s -cf test1/archive.tar -C in . >test1/c.out 2>test1/c.err", testprog));
	assertChdir("test1");
	assertEqualInt(0,
	    systemf("%s -xf archive.tar >c.out 2>c.err", testprog));
	assertIsSymlink("ld1", "d1", 1);
	assertIsSymlink("d1/link1", "file1", 0);
	assertIsSymlink("d1/linkX", "fileX", 0);
	assertIsSymlink("link2", "d1/file2", 0);
	assertIsSymlink("linkY", "d1/fileY", 0);
	assertChdir("..");

	/* Test 2: With -L, no symlink on command line. */
	assertMakeDir("test2", 0755);
	assertEqualInt(0,
	    systemf("%s -cf test2/archive.tar -L -C in . >test2/c.out 2>test2/c.err", testprog));
	assertChdir("test2");
	assertEqualInt(0,
	    systemf("%s -xf archive.tar >c.out 2>c.err", testprog));
	assertIsDir("ld1", umasked(0755));
	assertIsReg("d1/link1", umasked(0644));
	assertIsSymlink("d1/linkX", "fileX", 0);
	assertIsReg("link2", umasked(0644));
	assertIsSymlink("linkY", "d1/fileY", 0);
	assertChdir("..");

	/* Test 3: With -L, some symlinks on command line. */
	assertMakeDir("test3", 0755);
	assertEqualInt(0,
	    systemf("%s -cf test3/archive.tar -L -C in ld1 d1 link2 linkY >test2/c.out 2>test2/c.err", testprog));
	assertChdir("test3");
	assertEqualInt(0,
	    systemf("%s -xf archive.tar >c.out 2>c.err", testprog));
	assertIsDir("ld1", umasked(0755));
	assertIsReg("d1/link1", umasked(0644));
	assertIsSymlink("d1/linkX", "fileX", 0);
	assertIsReg("link2", umasked(0644));
	assertIsSymlink("linkY", "d1/fileY", 0);
	assertChdir("..");
}
