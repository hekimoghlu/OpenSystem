/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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

DEFINE_TEST(test_extract_tar_Z)
{
	const char *reffile = "test_extract.tar.Z";

	extract_reference_file(reffile);
	assertEqualInt(0, systemf("%s -xf %s >test.out 2>test.err",
	    testprog, reffile));

	assertFileExists("file1");
	assertTextFileContents("contents of file1.\n", "file1");
	assertFileExists("file2");
	assertTextFileContents("contents of file2.\n", "file2");
	assertEmptyFile("test.out");
	assertEmptyFile("test.err");
}
