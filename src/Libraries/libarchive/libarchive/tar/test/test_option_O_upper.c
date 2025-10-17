/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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

static const char *test4out[] = {"file1", "file2", NULL};
static const char *test5err[] = {"file1", "file2", NULL};

DEFINE_TEST(test_option_O_upper)
{
	assertMakeFile("file1", 0644, "file1");
	assertMakeFile("file2", 0644, "file2");
	assertEqualInt(0, systemf("%s -cf archive.tar file1 file2", testprog));

	/* Test 1: -x without -O */
	assertMakeDir("test1", 0755);
	assertChdir("test1");
	assertEqualInt(0,
	    systemf("%s -xf ../archive.tar >test.out 2>test.err", testprog));
	assertFileContents("file1", 5, "file1");
	assertFileContents("file2", 5, "file2");
	assertEmptyFile("test.out");
	assertEmptyFile("test.err");
	assertChdir("..");

	/* Test 2: -x with -O */
	assertMakeDir("test2", 0755);
	assertChdir("test2");
	assertEqualInt(0,
	    systemf("%s -xOf ../archive.tar file1 >test.out 2>test.err", testprog));
	assertFileNotExists("file1");
	assertFileNotExists("file2");
	assertFileContents("file1", 5, "test.out");
	assertEmptyFile("test.err");
	assertChdir("..");

	/* Test 3: -x with -O and multiple files */
	assertMakeDir("test3", 0755);
	assertChdir("test3");
	assertEqualInt(0,
	    systemf("%s -xOf ../archive.tar >test.out 2>test.err", testprog));
	assertFileNotExists("file1");
	assertFileNotExists("file2");
	assertFileContents("file1file2", 10, "test.out");
	assertEmptyFile("test.err");
	assertChdir("..");

	/* Test 4: -t without -O */
	assertMakeDir("test4", 0755);
	assertChdir("test4");
	assertEqualInt(0,
	    systemf("%s -tf ../archive.tar >test.out 2>test.err", testprog));
	assertFileContainsLinesAnyOrder("test.out", test4out);
	assertEmptyFile("test.err");
	assertChdir("..");

	/* Test 5: -t with -O */
	assertMakeDir("test5", 0755);
	assertChdir("test5");
	assertEqualInt(0,
	    systemf("%s -tOf ../archive.tar >test.out 2>test.err", testprog));
	assertEmptyFile("test.out");
	assertFileContainsLinesAnyOrder("test.err", test5err);
	assertChdir("..");
}
