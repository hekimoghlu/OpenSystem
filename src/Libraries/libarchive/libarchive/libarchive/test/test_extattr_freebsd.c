/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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

#if defined(__FreeBSD__) && __FreeBSD__ > 4
#include <sys/extattr.h>
#endif

/*
 * Verify extended attribute restore-to-disk.  This test is FreeBSD-specific.
 */

DEFINE_TEST(test_extattr_freebsd)
{
#if !defined(__FreeBSD__)
	skipping("FreeBSD-specific extattr restore test");
#elif __FreeBSD__ < 5
	skipping("extattr restore supported only on FreeBSD 5.0 and later");
#else
	char buff[64];
	const char *xname;
	const void *xval;
	size_t xsize;
	struct stat st;
	struct archive *a;
	struct archive_entry *ae;
	int n, fd;

	/*
	 * First, do a quick manual set/read of an extended attribute
	 * to verify that the local filesystem does support it.  If it
	 * doesn't, we'll simply skip the remaining tests.
	 */
	/* Create a test file and try to set an ACL on it. */
	fd = open("pretest", O_RDWR | O_CREAT, 0777);
	failure("Could not create test file?!");
	if (!assert(fd >= 0))
		return;

	errno = 0;
	n = extattr_set_fd(fd, EXTATTR_NAMESPACE_USER, "testattr", "1234", 4);
	if (n != 4 && errno == EOPNOTSUPP) {
		close(fd);
		skipping("extattr tests require that extattr support be enabled on the filesystem");
		return;
	}
	failure("extattr_set_fd(): errno=%d (%s)", errno, strerror(errno));
	assertEqualInt(4, n);
	close(fd);

	/* Create a write-to-disk object. */
	assert(NULL != (a = archive_write_disk_new()));
	archive_write_disk_set_options(a,
	    ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM | ARCHIVE_EXTRACT_XATTR);

	/* Populate an archive entry with an extended attribute. */
	ae = archive_entry_new();
	assert(ae != NULL);
	archive_entry_set_pathname(ae, "test0");
	archive_entry_set_mtime(ae, 123456, 7890);
	archive_entry_set_size(ae, 0);
	archive_entry_set_mode(ae, 0755);
	archive_entry_xattr_add_entry(ae, "user.foo", "12345", 5);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_finish_entry(a));
	archive_entry_free(ae);

	/* Another entry; similar but with mode = 0. */
	ae = archive_entry_new();
	assert(ae != NULL);
	archive_entry_set_pathname(ae, "test1");
	archive_entry_set_mtime(ae, 12345678, 7890);
	archive_entry_set_size(ae, 0);
	archive_entry_set_mode(ae, 0);
	archive_entry_xattr_add_entry(ae, "user.bar", "123456", 6);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Close the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));

	/* Verify the data on disk. */
	assertEqualInt(0, stat("test0", &st));
	assertEqualInt(st.st_mtime, 123456);
	assertEqualInt(st.st_mode & 0777, 0755);
	/* Verify extattr */
	n = extattr_get_file("test0", EXTATTR_NAMESPACE_USER,
	    "foo", buff, sizeof(buff));
	if (assertEqualInt(n, 5)) {
		buff[n] = '\0';
		assertEqualString(buff, "12345");
	}

	/* Verify the data on disk. */
	assertEqualInt(0, stat("test1", &st));
	assertEqualInt(st.st_mtime, 12345678);
	assertEqualInt(st.st_mode & 0777, 0);
	/*
	 * If we are not root, we have to make test1 user readable
	 * or extattr_get_file() will fail
	 */
	if (geteuid() != 0) {
		chmod("test1", S_IRUSR);
	}
	/* Verify extattr */
	n = extattr_get_file("test1", EXTATTR_NAMESPACE_USER,
	    "bar", buff, sizeof(buff));
	if (assertEqualInt(n, 6)) {
		buff[n] = '\0';
		assertEqualString(buff, "123456");
	}

	/* Use libarchive APIs to read the file back into an entry and
	 * verify that the extattr was read correctly. */
	assert((a = archive_read_disk_new()) != NULL);
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_pathname(ae, "test0");
	assertEqualInt(ARCHIVE_OK,
	    archive_read_disk_entry_from_file(a, ae, -1, NULL));
	assertEqualInt(1, archive_entry_xattr_reset(ae));
	assertEqualInt(ARCHIVE_OK,
	    archive_entry_xattr_next(ae, &xname, &xval, &xsize));
	assertEqualString(xname, "user.foo");
	assertEqualInt(xsize, 5);
	assertEqualMem(xval, "12345", xsize);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	archive_entry_free(ae);
#endif
}
