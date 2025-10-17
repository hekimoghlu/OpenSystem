/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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

DEFINE_TEST(test_print_longpath)
{
	const char *reffile = "test_print_longpath.tar.Z";
	char buff[2048];
	int i, j, k;

	/* Reference file has one entry "file" with a very old timestamp. */
	extract_reference_file(reffile);

	/* Build long path pattern. */
	memset(buff, 0, sizeof(buff));
	for (k = 0; k < 4; k++) {
		for (j = 0; j < k+1; j++) {
			for (i = 0; i < 10; i++)
				strncat(buff, "0123456789",
				    sizeof(buff) - strlen(buff) -1);
			strncat(buff, "/", sizeof(buff) - strlen(buff) -1);
		}
		strncat(buff, "\n", sizeof(buff) - strlen(buff) -1);
	}
	buff[sizeof(buff)-1] = '\0';

	assertEqualInt(0,
	    systemf("%s -tf %s >test.out 2>test.err", testprog, reffile));
	assertTextFileContents(buff, "test.out");
	assertEmptyFile("test.err");
}
