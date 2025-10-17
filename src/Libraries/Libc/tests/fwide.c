/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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

#include <stdio.h>
#include <wchar.h>

#include <darwintest.h>
#include <paths.h>

#define	FWIDE_TEST_PATH	_PATH_DEV "zero"

/*
 * These mostly test fgets/fgetwc, but they can point to issues with the
 * underlying orientation tracking.
 */

T_DECL(fwide_fgets,
    "Test that fgets(3) sets the stream orientation to byte-oriented")
{
	FILE *fp;
	char buf[1];

	fp = fopen(FWIDE_TEST_PATH, "r");
	T_WITH_ERRNO;
	T_ASSERT_NOTNULL(fp, NULL);

	T_ASSERT_EQ(fgets(&buf[0], sizeof(buf), fp), &buf[0], NULL);
	T_ASSERT_LT(fwide(fp, 0), 0, NULL);
}

T_DECL(fwide_fgetwc,
    "Test that fgetwc(3) sets the stream orientation to wide-oriented")
{
	FILE *fp;

	fp = fopen(FWIDE_TEST_PATH, "r");
	T_WITH_ERRNO;
	T_ASSERT_NOTNULL(fp, NULL);

	T_ASSERT_EQ(fgetwc(fp), 0, NULL);
	T_ASSERT_GT(fwide(fp, 0), 0, NULL);
}

T_DECL(fwide_nop, "Test that fwide(3) is a nop after orientation is set")
{
	FILE *fp;

	fp = fopen(FWIDE_TEST_PATH, "r");
	T_WITH_ERRNO;
	T_ASSERT_NOTNULL(fp, NULL);

	T_ASSERT_EQ(fwide(fp, 0), 0, NULL);
	T_ASSERT_EQ(fgetwc(fp), 0, NULL);
	T_ASSERT_GT(fwide(fp, -1), 0, NULL);
}
