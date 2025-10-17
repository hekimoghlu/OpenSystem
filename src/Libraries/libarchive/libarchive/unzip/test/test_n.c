/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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

/* Test n arg - don't overrite existing files */
DEFINE_TEST(test_n)
{
	const char *reffile = "test_basic.zip";
	int r;

	assertMakeDir("test_basic", 0755);
	assertMakeFile("test_basic/a", 0644, "orig a\n");
	assertMakeFile("test_basic/b", 0644, "orig b\n");

	extract_reference_file(reffile);
	r = systemf("%s -n %s >test.out 2>test.err", testprog, reffile);
	assertEqualInt(0, r);
	assertNonEmptyFile("test.out");
	assertEmptyFile("test.err");

	assertTextFileContents("orig a\n", "test_basic/a");
	assertTextFileContents("orig b\n", "test_basic/b");
	assertTextFileContents("contents c\n", "test_basic/c");
	assertTextFileContents("contents CAPS\n", "test_basic/CAPS");
}
