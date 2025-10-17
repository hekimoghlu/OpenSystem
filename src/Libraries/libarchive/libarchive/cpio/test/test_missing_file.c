/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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

DEFINE_TEST(test_missing_file)
{
	int r;

	assertMakeFile("file1", 0644, "file1");
	assertMakeFile("file2", 0644, "file2");

	assertMakeFile("filelist1", 0644, "file1\nfile2\n");
	r = systemf("%s -o <filelist1 >stdout1 2>stderr1", testprog);
	assertEqualInt(r, 0);
	assertTextFileContents("1 block\n", "stderr1");

	assertMakeFile("filelist2", 0644, "file1\nfile2\nfile3\n");
	r = systemf("%s -o <filelist2 >stdout2 2>stderr2", testprog);
	assert(r != 0);

	assertMakeFile("filelist3", 0644, "");
	r = systemf("%s -o <filelist3 >stdout3 2>stderr3", testprog);
	assertEqualInt(r, 0);
	assertTextFileContents("1 block\n", "stderr3");

	assertMakeFile("filelist4", 0644, "file3\n");
	r = systemf("%s -o <filelist4 >stdout4 2>stderr4", testprog);
	assert(r != 0);
}
