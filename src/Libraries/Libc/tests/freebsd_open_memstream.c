/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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

#include <err.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include <darwintest.h>

static char *buf;
static size_t len;

static void
assert_stream(const char *contents)
{
	T_EXPECT_EQ(strlen(contents), len,
			"bad length %zd for \"%s\"\n", len, contents);
    T_EXPECT_EQ(strncmp(buf, contents, strlen(contents)), 0,
			"bad buffer \"%s\" for \"%s\"\n", buf, contents);
}

T_DECL(freebsd_open_memstream_open_group_test, "")
{
	FILE *fp;
	off_t eob;

	fp = open_memstream(&buf, &len);
	T_ASSERT_NOTNULL(fp, "open_memstream failed");

	fprintf(fp, "hello my world");
	fflush(fp);
	assert_stream("hello my world");
	eob = ftello(fp);
	rewind(fp);
	fprintf(fp, "good-bye");
	fseeko(fp, eob, SEEK_SET);
	fclose(fp);
	assert_stream("good-bye world");
	free(buf);
}

T_DECL(freebsd_open_memstream_simple_tests, "")
{
	static const char zerobuf[] =
	    { 'f', 'o', 'o', 0, 0, 0, 0, 'b', 'a', 'r', 0 };
	char c;
	FILE *fp;

	fp = open_memstream(&buf, NULL);
	T_ASSERT_NULL(fp, "open_memstream did not fail");
	T_ASSERT_EQ(errno, EINVAL, "open_memstream didn't fail with EINVAL");
	fp = open_memstream(NULL, &len);
	T_ASSERT_NULL(fp, "open_memstream did not fail");
	T_ASSERT_EQ(errno, EINVAL, "open_memstream didn't fail with EINVAL");
	fp = open_memstream(&buf, &len);
	T_ASSERT_NOTNULL(fp, "open_memstream failed, errno=%d", errno);
	fflush(fp);
	assert_stream("");
	if (fwide(fp, 0) >= 0)
		printf("stream is not byte-oriented\n");

	fprintf(fp, "fo");
	fflush(fp);
	assert_stream("fo");
	fputc('o', fp);
	fflush(fp);
	assert_stream("foo");
	rewind(fp);
	fflush(fp);
	assert_stream("");
	fseek(fp, 0, SEEK_END);
	fflush(fp);
	assert_stream("foo");

	/*
	 * Test seeking out past the current end.  Should zero-fill the
	 * intermediate area.
	 */
	fseek(fp, 4, SEEK_END);
	fprintf(fp, "bar");
	fflush(fp);

	/*
	 * Can't use assert_stream() here since this should contain
	 * embedded null characters.
	 */
	if (len != 10)
		printf("bad length %zd for zero-fill test\n", len);
	else if (memcmp(buf, zerobuf, sizeof(zerobuf)) != 0)
		printf("bad buffer for zero-fill test\n");

	fseek(fp, 3, SEEK_SET);
	fprintf(fp, " in ");
	fflush(fp);
	assert_stream("foo in ");
	fseek(fp, 0, SEEK_END);
	fflush(fp);
	assert_stream("foo in bar");

	rewind(fp);
	if (fread(&c, sizeof(c), 1, fp) != 0)
		printf("fread did not fail\n");
	else if (!ferror(fp))
		printf("error indicator not set after fread\n");
	else
		clearerr(fp);

	fseek(fp, 4, SEEK_SET);
	fprintf(fp, "bar baz");
	fclose(fp);
	assert_stream("foo bar baz");
	free(buf);
}

T_DECL(freebsd_open_memstream_seek_tests, "")
{
	FILE *fp;

	fp = open_memstream(&buf, &len);
	T_ASSERT_NOTNULL(fp, "open_memstream failed: %d", errno);

#define SEEK_FAIL(offset, whence, error) do {			\
	errno = 0;						\
	T_ASSERT_NE(fseeko(fp, (offset), (whence)), 0,	\
	    "fseeko(%s, %s) did not fail, set pos to %jd",	\
	    __STRING(offset), __STRING(whence), \
	    (intmax_t)ftello(fp));				\
	T_ASSERT_EQ(errno, (error),			\
	    "fseeko(%s, %s) failed with %d rather than %s",	\
	    __STRING(offset), __STRING(whence),	errno,		\
	    __STRING(error));					\
} while (0)

#define SEEK_OK(offset, whence, result) do {			\
	T_ASSERT_EQ(fseeko(fp, (offset), (whence)), 0,	\
	    "fseeko(%s, %s) failed: %s",			\
	    __STRING(offset), __STRING(whence), strerror(errno)); \
	T_ASSERT_EQ(ftello(fp), (off_t)(result),			\
	    "fseeko(%s, %s) seeked to %jd rather than %s",	\
	    __STRING(offset), __STRING(whence),			\
	    (intmax_t)ftello(fp), __STRING(result));		\
} while (0)

	SEEK_FAIL(-1, SEEK_SET, EINVAL);
	SEEK_FAIL(-1, SEEK_CUR, EINVAL);
	SEEK_FAIL(-1, SEEK_END, EINVAL);
	fprintf(fp, "foo");
	SEEK_OK(-1, SEEK_CUR, 2);
	SEEK_OK(0, SEEK_SET, 0);
	SEEK_OK(-1, SEEK_END, 2);
	SEEK_OK(OFF_MAX - 1, SEEK_SET, OFF_MAX - 1);
	SEEK_FAIL(2, SEEK_CUR, EOVERFLOW);
	fclose(fp);
}
