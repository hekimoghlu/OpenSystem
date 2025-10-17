/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
 * As reported by Bernd Walter:  Some people are in the habit of
 * using "find -d" to generate a list for cpio -p because that
 * copies the top-level dir last, which preserves owner and mode
 * information.  That's not necessary for bsdcpio (libarchive defers
 * restoring directory information), but bsdcpio should still generate
 * the correct results with this usage.
 */

DEFINE_TEST(test_passthrough_reverse)
{
	int r;
	FILE *filelist;

	assertUmask(0);

	/*
	 * Create an assortment of files on disk.
	 */
	filelist = fopen("filelist", "w");

	/* Directory. */
	assertMakeDir("dir", 0743);

	/* File with 10 bytes content. */
	assertMakeFile("dir/file", 0644, "1234567890");
	fprintf(filelist, "dir/file\n");

	/* Write dir last. */
	fprintf(filelist, "dir\n");

	/* All done. */
	fclose(filelist);


	/*
	 * Use cpio passthrough mode to copy files to another directory.
	 */
	r = systemf("%s -pdvm out <filelist >stdout 2>stderr", testprog);
	failure("Error invoking %s -pd out", testprog);
	assertEqualInt(r, 0);

	assertChdir("out");

	/* Verify stderr and stdout. */
	assertTextFileContents("out/dir/file\nout/dir\n1 block\n",
	    "../stderr");
	assertEmptyFile("../stdout");

	/* dir */
	assertIsDir("dir", 0743);


	/* Regular file. */
	assertIsReg("dir/file", 0644);
	assertFileSize("dir/file", 10);
	assertFileNLinks("dir/file", 1);
}
