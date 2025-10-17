/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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
	const char * invalid_stderr[] = { "INTERNAL ERROR", NULL };
	assertMakeFile("file1", 0644, "file1");
	assertMakeFile("file2", 0644, "file2");
	assert(0 == systemf("%s -cf archive.tar file1 file2 2>stderr1", testprog));
	assertEmptyFile("stderr1");
	assert(0 != systemf("%s -cf archive.tar file1 file2 file3 2>stderr2", testprog));
	assertFileContainsNoInvalidStrings("stderr2", invalid_stderr);
	assert(0 != systemf("%s -cf archive.tar 2>stderr3", testprog));
	assertFileContainsNoInvalidStrings("stderr3", invalid_stderr);
	assert(0 != systemf("%s -cf archive.tar file3 file4 2>stderr4", testprog));
	assertFileContainsNoInvalidStrings("stderr4", invalid_stderr);
}
