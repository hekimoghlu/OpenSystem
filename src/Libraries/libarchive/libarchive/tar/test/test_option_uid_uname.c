/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

DEFINE_TEST(test_option_uid_uname)
{
	char *reference, *data;
	size_t s;

	assertUmask(0);
	assertMakeFile("file", 0644, "1234567890");

	/* Create archive with no special options. */
	failure("Error invoking %s c", testprog);
	assertEqualInt(0,
	    systemf("%s cf archive1 --format=ustar file >stdout1.txt 2>stderr1.txt",
		testprog));
	assertEmptyFile("stdout1.txt");
	assertEmptyFile("stderr1.txt");
	reference = slurpfile(&s, "archive1");

	/* Again with both --uid and --uname */
	failure("Error invoking %s c", testprog);
	assertEqualInt(0,
	    systemf("%s cf archive2 --uid=65123 --uname=foofoofoo --format=ustar file >stdout2.txt 2>stderr2.txt",
		testprog));
	assertEmptyFile("stdout2.txt");
	assertEmptyFile("stderr2.txt");
	data = slurpfile(&s, "archive2");
	/* Should force uid and uname fields in ustar header. */
	assertEqualMem(data + 108, "177143 \0", 8);
	assertEqualMem(data + 265, "foofoofoo\0", 10);
	free(data);

	/* Again with just --uid */
	failure("Error invoking %s c", testprog);
	assertEqualInt(0,
	    systemf("%s cf archive3 --uid=65123 --format=ustar file >stdout3.txt 2>stderr3.txt",
		testprog));
	assertEmptyFile("stdout3.txt");
	assertEmptyFile("stderr3.txt");
	data = slurpfile(&s, "archive3");
	assertEqualMem(data + 108, "177143 \0", 8);
	/* Uname field in ustar header should be empty. */
	assertEqualMem(data + 265, "\0", 1);
	free(data);

	/* Again with just --uname */
	failure("Error invoking %s c", testprog);
	assertEqualInt(0,
	    systemf("%s cf archive4 --uname=foofoofoo --format=ustar file >stdout4.txt 2>stderr4.txt",
		testprog));
	assertEmptyFile("stdout4.txt");
	assertEmptyFile("stderr4.txt");
	data = slurpfile(&s, "archive4");
	/* Uid should be unchanged from original reference. */
	assertEqualMem(data + 108, reference + 108, 8);
	assertEqualMem(data + 265, "foofoofoo\0", 10);
	free(data);

	free(reference);
}
