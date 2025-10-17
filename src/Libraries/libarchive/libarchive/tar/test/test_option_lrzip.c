/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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

DEFINE_TEST(test_option_lrzip)
{
	char *p;
	size_t s;

	if (!canLrzip()) {
		skipping("lrzip is not supported on this platform");
		return;
	}

	/* Create a file. */
	assertMakeFile("f", 0644, "a");

	/* Archive it with lrzip compression. */
	assertEqualInt(0,
	    systemf("%s -cf - --lrzip f >archive.out 2>archive.err",
	    testprog));
	p = slurpfile(&s, "archive.err");
	p[s] = '\0';
	free(p);
	/* Check that the archive file has an lzma signature. */
	p = slurpfile(&s, "archive.out");
	assert(s > 2);
	assertEqualMem(p, "LRZI\x00", 5);
	free(p);
}
