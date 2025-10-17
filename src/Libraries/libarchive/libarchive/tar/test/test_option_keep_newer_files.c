/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

DEFINE_TEST(test_option_keep_newer_files)
{
	const char *reffile = "test_option_keep_newer_files.tar.Z";

	/* Reference file has one entry "file" with a very old timestamp. */
	extract_reference_file(reffile);

	/* Test 1: Without --keep-newer-files */
	assertMakeDir("test1", 0755);
	assertChdir("test1");
	assertMakeFile("file", 0644, "new");
	assertEqualInt(0,
	    systemf("%s -xf ../%s >test.out 2>test.err", testprog, reffile));
	assertFileContents("old\n", 4, "file");
	assertEmptyFile("test.out");
	assertEmptyFile("test.err");
	assertChdir("..");

	/* Test 2: With --keep-newer-files */
	assertMakeDir("test2", 0755);
	assertChdir("test2");
	assertMakeFile("file", 0644, "new");
	assertEqualInt(0,
	    systemf("%s -xf ../%s --keep-newer-files >test.out 2>test.err", testprog, reffile));
	assertFileContents("new", 3, "file");
	assertEmptyFile("test.out");
	assertEmptyFile("test.err");
	assertChdir("..");
}
