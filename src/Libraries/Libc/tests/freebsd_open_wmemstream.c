/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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

static wchar_t *buf;
static size_t len;

static void
assert_stream(const wchar_t *contents)
{
	if (wcslen(contents) != len)
		printf("bad length %zd for \"%ls\"\n", len, contents);
	else if (wcsncmp(buf, contents, wcslen(contents)) != 0)
		printf("bad buffer \"%ls\" for \"%ls\"\n", buf, contents);
}

T_DECL(freebsd_open_wmemstream_open_group_test, "")
{
	FILE *fp;
	off_t eob;

	fp = open_wmemstream(&buf, &len);
	T_ASSERT_NOTNULL(fp, "open_wmemstream failed");

	fwprintf(fp, L"hello my world");
	fflush(fp);
	assert_stream(L"hello my world");
	eob = ftello(fp);
	rewind(fp);
	fwprintf(fp, L"good-bye");
	fseeko(fp, eob, SEEK_SET);
	fclose(fp);
	assert_stream(L"good-bye world");
	free(buf);
}

T_DECL(freebsd_open_wmemstream_simple_tests, "")
{
	static const wchar_t zerobuf[] =
	    { L'f', L'o', L'o', 0, 0, 0, 0, L'b', L'a', L'r', 0 };
	wchar_t c;
	FILE *fp;

	fp = open_wmemstream(&buf, NULL);
	T_ASSERT_NULL(fp, "open_wmemstream did not fail");
	T_ASSERT_EQ(errno, EINVAL, "open_wmemstream didn't fail with EINVAL");
	fp = open_wmemstream(NULL, &len);
	T_ASSERT_NULL(fp, "open_wmemstream did not fail");
	T_ASSERT_EQ(errno, EINVAL, "open_wmemstream didn't fail with EINVAL");
	fp = open_wmemstream(&buf, &len);
	T_ASSERT_NOTNULL(fp, "open_memstream failed, errno=%d", errno);
	fflush(fp);
	assert_stream(L"");
	if (fwide(fp, 0) <= 0)
		printf("stream is not wide-oriented\n");

	fwprintf(fp, L"fo");
	fflush(fp);
	assert_stream(L"fo");
	fputwc(L'o', fp);
	fflush(fp);
	assert_stream(L"foo");
	rewind(fp);
	fflush(fp);
	assert_stream(L"");
	fseek(fp, 0, SEEK_END);
	fflush(fp);
	assert_stream(L"foo");

	/*
	 * Test seeking out past the current end.  Should zero-fill the
	 * intermediate area.
	 */
	fseek(fp, 4, SEEK_END);
	fwprintf(fp, L"bar");
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
	fwprintf(fp, L" in ");
	fflush(fp);
	assert_stream(L"foo in ");
	fseek(fp, 0, SEEK_END);
	fflush(fp);
	assert_stream(L"foo in bar");

	rewind(fp);
	if (fread(&c, sizeof(c), 1, fp) != 0)
		printf("fread did not fail\n");
	else if (!ferror(fp))
		printf("error indicator not set after fread\n");
	else
		clearerr(fp);

	fseek(fp, 4, SEEK_SET);
	fwprintf(fp, L"bar baz");
	fclose(fp);
	assert_stream(L"foo bar baz");
	free(buf);
}

T_DECL(freebsd_open_wmemstream_seek_tests, "")
{
	FILE *fp;

	fp = open_wmemstream(&buf, &len);
	T_ASSERT_NOTNULL(fp, "open_wmemstream failed, errno=%d", errno);

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
	fwprintf(fp, L"foo");
	SEEK_OK(-1, SEEK_CUR, 2);
	SEEK_OK(0, SEEK_SET, 0);
	SEEK_OK(-1, SEEK_END, 2);
	SEEK_OK(OFF_MAX - 1, SEEK_SET, OFF_MAX - 1);
	SEEK_FAIL(2, SEEK_CUR, EOVERFLOW);
	fclose(fp);
}
