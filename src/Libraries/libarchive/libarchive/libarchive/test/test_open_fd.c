/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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

#if defined(_WIN32) && !defined(__CYGWIN__)
#define open _open
#if !defined(__BORLANDC__)
#ifdef lseek
#undef lseek
#endif
#define lseek _lseek
#endif
#define close _close
#endif

DEFINE_TEST(test_open_fd)
{
	char buff[64];
	struct archive_entry *ae;
	struct archive *a;
	int fd;
	const char *skip_open_fd_err_test;

#if defined(__BORLANDC__)
	fd = open("test.tar", O_RDWR | O_CREAT | O_BINARY);
#else
	fd = open("test.tar", O_RDWR | O_CREAT | O_BINARY, 0777);
#endif
	assert(fd >= 0);
	if (fd < 0)
		return;

	/* Write an archive through this fd. */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_add_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_open_fd(a, fd));

	/*
	 * Write a file to it.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_mtime(ae, 1, 0);
	archive_entry_copy_pathname(ae, "file");
	archive_entry_set_mode(ae, S_IFREG | 0755);
	archive_entry_set_size(ae, 8);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, 8, archive_write_data(a, "12345678", 9));

	/*
	 * Write a second file to it.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "file2");
	archive_entry_set_mode(ae, S_IFREG | 0755);
	archive_entry_set_size(ae, 819200);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Close out the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/*
	 * Now, read the data back.
	 */
	assert(lseek(fd, 0, SEEK_SET) == 0);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_fd(a, fd, 512));

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(1, archive_entry_mtime(ae));
	assertEqualInt(0, archive_entry_mtime_nsec(ae));
	assertEqualInt(0, archive_entry_atime(ae));
	assertEqualInt(0, archive_entry_ctime(ae));
	assertEqualString("file", archive_entry_pathname(ae));
	assert((S_IFREG | 0755) == archive_entry_mode(ae));
	assertEqualInt(8, archive_entry_size(ae));
	assertEqualIntA(a, 8, archive_read_data(a, buff, 10));
	assertEqualMem(buff, "12345678", 8);

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("file2", archive_entry_pathname(ae));
	assert((S_IFREG | 0755) == archive_entry_mode(ae));
	assertEqualInt(819200, archive_entry_size(ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_data_skip(a));

	/* Verify the end of the archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	close(fd);

	skip_open_fd_err_test = getenv("SKIP_OPEN_FD_ERR_TEST");
	if(skip_open_fd_err_test == NULL) {
		/*
		 * Verify some of the error handling.
		 */
		assert((a = archive_read_new()) != NULL);
		assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
		assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
		/* FD 100 shouldn't be open. */
		assertEqualIntA(a, ARCHIVE_FATAL,
		archive_read_open_fd(a, 100, 512));
		assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	}
}
