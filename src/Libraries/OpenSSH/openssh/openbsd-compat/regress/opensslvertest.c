/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#include "includes.h"

#include <stdio.h>
#include <stdlib.h>

int ssh_compatible_openssl(long, long);

struct version_test {
	long headerver;
	long libver;
	int result;
} version_tests[] = {
	/* built with 1.0.1b release headers */
	{ 0x1000101fL, 0x1000101fL, 1},/* exact match */
	{ 0x1000101fL, 0x1000102fL, 1},	/* newer library patch version: ok */
	{ 0x1000101fL, 0x1000100fL, 1},	/* older library patch version: ok */
	{ 0x1000101fL, 0x1000201fL, 1},	/* newer library fix version: ok */
	{ 0x1000101fL, 0x1000001fL, 0},	/* older library fix version: NO */
	{ 0x1000101fL, 0x1010101fL, 0},	/* newer library minor version: NO */
	{ 0x1000101fL, 0x0000101fL, 0},	/* older library major version: NO */
	{ 0x1000101fL, 0x2000101fL, 0},	/* newer library major version: NO */

	/* built with 1.1.1b release headers */
	{ 0x1010101fL, 0x1010101fL, 1},/* exact match */
	{ 0x1010101fL, 0x1010102fL, 1},	/* newer library patch version: ok */
	{ 0x1010101fL, 0x1010100fL, 1},	/* older library patch version: ok */
	{ 0x1010101fL, 0x1010201fL, 1},	/* newer library fix version: ok */
	{ 0x1010101fL, 0x1010001fL, 0},	/* older library fix version: NO */
	{ 0x1010101fL, 0x1020001fL, 0},	/* newer library minor version: NO */
	{ 0x1010101fL, 0x0010101fL, 0},	/* older library major version: NO */
	{ 0x1010101fL, 0x2010101fL, 0},	/* newer library major version: NO */

	/* built with 3.0.1 release headers */
	{ 0x3010101fL, 0x3010101fL, 1},/* exact match */
	{ 0x3010101fL, 0x3010102fL, 1},	/* newer library patch version: ok */
	{ 0x3010101fL, 0x3010100fL, 1},	/* older library patch version: ok */
	{ 0x3010101fL, 0x3010201fL, 1},	/* newer library fix version: ok */
	{ 0x3010101fL, 0x3010001fL, 1},	/* older library fix version: ok */
	{ 0x3010101fL, 0x3020001fL, 1},	/* newer library minor version: ok */
	{ 0x3010101fL, 0x1010101fL, 0},	/* older library major version: NO */
	{ 0x3010101fL, 0x4010101fL, 0},	/* newer library major version: NO */
};

void
fail(long hver, long lver, int result)
{
	fprintf(stderr, "opensslver: header %lx library %lx != %d \n", hver, lver, result);
	exit(1);
}

int
main(void)
{
#ifdef WITH_OPENSSL
	unsigned int i;
	int res;
	long hver, lver;

	for (i = 0; i < sizeof(version_tests) / sizeof(version_tests[0]); i++) {
		hver = version_tests[i].headerver;
		lver = version_tests[i].libver;
		res = version_tests[i].result;
		if (ssh_compatible_openssl(hver, lver) != res)
			fail(hver, lver, res);
	}
#endif
	exit(0);
}
