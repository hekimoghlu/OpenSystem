/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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

/*
 * Verify that "cpio -p .." works.
 */

DEFINE_TEST(test_passthrough_dotdot)
{
	int r;
	FILE *filelist;

	assertUmask(0);

	/*
	 * Create an assortment of files on disk.
	 */
	filelist = fopen("filelist", "w");

	/* Directory. */
	assertMakeDir("dir", 0755);
	assertChdir("dir");

	fprintf(filelist, ".\n");

	/* File with 10 bytes content. */
	assertMakeFile("file", 0642, "1234567890");
	fprintf(filelist, "file\n");

	/* All done. */
	fclose(filelist);


	/*
	 * Use cpio passthrough mode to copy files to another directory.
	 */
	r = systemf("%s -pdvm .. <../filelist >../stdout 2>../stderr",
	    testprog);
	failure("Error invoking %s -pd ..", testprog);
	assertEqualInt(r, 0);

	assertChdir("..");

	/* Verify stderr and stdout. */
	assertTextFileContents("../.\n../file\n1 block\n", "stderr");
	assertEmptyFile("stdout");

	/* Regular file. */
	assertIsReg("file", 0642);
	assertFileSize("file", 10);
	assertFileNLinks("file", 1);
}
