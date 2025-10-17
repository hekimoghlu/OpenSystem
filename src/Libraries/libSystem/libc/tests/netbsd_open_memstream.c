/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#include <sys/cdefs.h>
__RCSID("$NetBSD: t_open_memstream.c,v 1.2 2014/10/19 11:17:43 justin Exp $");

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <darwintest.h>

#define OFFSET 16384

static const char start[] = "start";
static const char hello[] = "hello";

T_DECL(netbsd_open_memstream_test_open_memstream, "")
{
	FILE	*fp;
	char	*buf = (char *)0xff;
	size_t	 size = 0;
	off_t	 off;
	int	 i;

	fp = open_memstream(&buf, &size);
	T_ASSERT_NOTNULL(fp, NULL);

	off = ftello(fp);
	T_EXPECT_EQ(off, 0LL, NULL);

	T_EXPECT_POSIX_ZERO(fflush(fp), NULL);
	T_EXPECT_EQ(size, 0UL, NULL);
	T_EXPECT_NE((void*)buf, (void *)0xff, NULL);
	T_EXPECT_EQ(fseek(fp, -6, SEEK_SET), -1, NULL);
	T_EXPECT_POSIX_ZERO(fseek(fp, OFFSET, SEEK_SET), NULL);
	T_EXPECT_NE(fprintf(fp, hello), EOF, NULL);
	T_EXPECT_NE(fflush(fp), EOF, NULL);
	T_EXPECT_EQ(size, OFFSET + sizeof(hello)-1, NULL);
	T_EXPECT_POSIX_ZERO(fseek(fp, 0, SEEK_SET), NULL);
	T_EXPECT_NE(fprintf(fp, start), EOF, NULL);
	T_EXPECT_NE(fflush(fp), EOF, NULL);
	T_EXPECT_EQ(size, sizeof(start)-1, NULL);

	/* Needed for sparse files */
	T_EXPECT_EQ(strncmp(buf, start, sizeof(start)-1), 0, NULL);
	for (i = sizeof(start)-1; i < OFFSET; i++)
		T_EXPECT_EQ(buf[i], '\0', NULL);

	T_EXPECT_EQ(memcmp(buf + OFFSET, hello, sizeof(hello)-1), 0, NULL);

	/* verify that simply seeking past the end doesn't increase the size */
	T_EXPECT_POSIX_ZERO(fseek(fp, 100, SEEK_END), NULL);
	T_EXPECT_NE(fflush(fp), EOF, NULL);
	T_EXPECT_EQ(size, OFFSET + sizeof(hello)-1, NULL);
	T_EXPECT_POSIX_ZERO(fseek(fp, 8, SEEK_SET), NULL);
	T_EXPECT_EQ(ftell(fp), 8L, NULL);

	/* Try to seek backward */
	T_EXPECT_POSIX_ZERO(fseek(fp, -1, SEEK_CUR), NULL);
	T_EXPECT_EQ(ftell(fp), 7L, NULL);
	T_EXPECT_POSIX_ZERO(fseek(fp, 5, SEEK_CUR), NULL);
	T_EXPECT_NE(fclose(fp), EOF, NULL);
	T_EXPECT_EQ(size, 12UL, NULL);

	free(buf);
}
