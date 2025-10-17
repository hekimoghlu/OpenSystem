/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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

/* Test d arg - extract to target dir - before zipfile argument */
DEFINE_TEST(test_d_before_zipfile)
{
	const char *reffile = "test_basic.zip";
	int r;

	extract_reference_file(reffile);
	r = systemf("%s -d foobar %s >test.out 2>test.err", testprog, reffile);
	assertEqualInt(0, r);
	assertNonEmptyFile("test.out");
	assertEmptyFile("test.err");

	assertTextFileContents("contents a\n", "foobar/test_basic/a");
	assertTextFileContents("contents b\n", "foobar/test_basic/b");
	assertTextFileContents("contents c\n", "foobar/test_basic/c");
	assertTextFileContents("contents CAPS\n", "foobar/test_basic/CAPS");
}

/* Test d arg - extract to target dir - after zipfile argument */
DEFINE_TEST(test_d_after_zipfile)
{
	const char *reffile = "test_basic.zip";
	int r;

	extract_reference_file(reffile);
	r = systemf("%s %s -d foobar >test.out 2>test.err", testprog, reffile);
	assertEqualInt(0, r);
	assertNonEmptyFile("test.out");
	assertEmptyFile("test.err");

	assertTextFileContents("contents a\n", "foobar/test_basic/a");
	assertTextFileContents("contents b\n", "foobar/test_basic/b");
	assertTextFileContents("contents c\n", "foobar/test_basic/c");
	assertTextFileContents("contents CAPS\n", "foobar/test_basic/CAPS");
}
