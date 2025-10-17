/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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

DEFINE_TEST(test_option_safe_writes)
{
	/* Create files */
	assertMakeDir("in", 0755);
	assertEqualInt(0, chdir("in"));
	assertMakeFile("f", 0644, "a");
	assertMakeFile("fh", 0644, "b");
	assertMakeFile("d", 0644, "c");
	assertMakeFile("fs", 0644, "d");
	assertMakeFile("ds", 0644, "e");
	assertEqualInt(0, chdir(".."));

	/* Tar files up */
	assertEqualInt(0,
	    systemf("%s -c -C in -f t.tar f fh d fs ds "
	    ">pack.out 2>pack.err", testprog));

        /* Verify that nothing went to stdout or stderr. */
        assertEmptyFile("pack.err");
        assertEmptyFile("pack.out");

	/* Create various objects */
	assertMakeDir("out", 0755);
	assertEqualInt(0, chdir("out"));
	assertMakeFile("f", 0644, "a");
	assertMakeHardlink("fh", "f");
	assertMakeDir("d", 0755);
	if (canSymlink()) {
		assertMakeSymlink("fs", "f", 0);
		assertMakeSymlink("ds", "d", 1);
	}
	assertEqualInt(0, chdir(".."));

	/* Extract created archive with safe writes */
	assertEqualInt(0,
	    systemf("%s -x -C out --safe-writes -f t.tar "
	    ">unpack.out 2>unpack.err", testprog));

        /* Verify that nothing went to stdout or stderr. */
        assertEmptyFile("unpack.err");
        assertEmptyFile("unpack.out");

	/* Verify that files were overwritten properly */
	assertEqualInt(0, chdir("out"));
	assertTextFileContents("a","f");
	assertTextFileContents("b","fh");
	assertTextFileContents("c","d");
	assertTextFileContents("d","fs");
	assertTextFileContents("e","ds");
}
