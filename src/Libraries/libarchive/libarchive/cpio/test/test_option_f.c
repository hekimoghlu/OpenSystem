/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
 * Unpack the archive in a new dir.
 */
static void
unpack(const char *dirname, const char *option)
{
	int r;

	assertMakeDir(dirname, 0755);
	assertChdir(dirname);
	extract_reference_file("test_option_f.cpio");
	r = systemf("%s -i %s < test_option_f.cpio > copy-no-a.out 2>copy-no-a.err", testprog, option);
	assertEqualInt(0, r);
	assertChdir("..");
}

DEFINE_TEST(test_option_f)
{
	/* Calibrate:  No -f option, so everything should be extracted. */
	unpack("t0", "--no-preserve-owner");
	assertFileExists("t0/a123");
	assertFileExists("t0/a234");
	assertFileExists("t0/b123");
	assertFileExists("t0/b234");

	/* Don't extract 'a*' files. */
#if defined(_WIN32) && !defined(__CYGWIN__)
	/* Single quotes isn't used by command.exe. */
	unpack("t1", "--no-preserve-owner -f a*");
#else
	unpack("t1", "--no-preserve-owner -f 'a*'");
#endif
	assertFileNotExists("t1/a123");
	assertFileNotExists("t1/a234");
	assertFileExists("t1/b123");
	assertFileExists("t1/b234");

	/* Don't extract 'b*' files. */
#if defined(_WIN32) && !defined(__CYGWIN__)
	/* Single quotes isn't used by command.exe. */
	unpack("t2", "--no-preserve-owner -f b*");
#else
	unpack("t2", "--no-preserve-owner -f 'b*'");
#endif
	assertFileExists("t2/a123");
	assertFileExists("t2/a234");
	assertFileNotExists("t2/b123");
	assertFileNotExists("t2/b234");
}
