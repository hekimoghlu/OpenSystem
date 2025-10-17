/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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

DEFINE_TEST(test_option_m)
{
	int r;

	/*
	 * The reference archive has one file with an mtime in 1970, 1
	 * second after the start of the epoch.
	 */

	/* Restored without -m, the result should have a current mtime. */
	assertMakeDir("without-m", 0755);
	assertChdir("without-m");
	extract_reference_file("test_option_m.cpio");
	r = systemf("%s --no-preserve-owner -i < test_option_m.cpio >out 2>err", testprog);
	assertEqualInt(r, 0);
	assertEmptyFile("out");
	assertTextFileContents("1 block\n", "err");
	/* Should have been created within the last few seconds. */
	assertFileMtimeRecent("file");

	/* With -m, it should have an mtime in 1970. */
	assertChdir("..");
	assertMakeDir("with-m", 0755);
	assertChdir("with-m");
	extract_reference_file("test_option_m.cpio");
	r = systemf("%s --no-preserve-owner -im < test_option_m.cpio >out 2>err", testprog);
	assertEqualInt(r, 0);
	assertEmptyFile("out");
	assertTextFileContents("1 block\n", "err");
	/*
	 * mtime in reference archive is '1' == 1 second after
	 * midnight Jan 1, 1970 UTC.
	 */
	assertFileMtime("file", 1, 0);
}
