/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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

#define UMASK 022

/*
 * Exercise security checks that should prevent certain
 * writes.
 */

DEFINE_TEST(test_write_disk_secure)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	skipping("archive_write_disk security checks not supported on Windows");
#else
	struct archive *a;
	struct archive_entry *ae;
	struct stat st;
#if defined(HAVE_LCHMOD) && defined(HAVE_SYMLINK) && \
    defined(S_IRUSR) && defined(S_IWUSR) && defined(S_IXUSR)
	int working_lchmod;
#endif

	/* Start with a known umask. */
	assertUmask(UMASK);

	/* Create an archive_write_disk object. */
	assert((a = archive_write_disk_new()) != NULL);

	/* Write a regular dir to it. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "dir");
	archive_entry_set_mode(ae, S_IFDIR | 0777);
	assert(0 == archive_write_header(a, ae));
	archive_entry_free(ae);
	assert(0 == archive_write_finish_entry(a));

	/* Write a symlink to the dir above. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir");
	archive_entry_set_mode(ae, S_IFLNK | 0777);
	archive_entry_set_symlink(ae, "dir");
	archive_write_disk_set_options(a, 0);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));

	/*
	 * Without security checks, we should be able to
	 * extract a file through the link.
	 */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir/filea");
	archive_entry_set_mode(ae, S_IFREG | 0777);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));

	/* But with security checks enabled, this should fail. */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir/fileb");
	archive_entry_set_mode(ae, S_IFREG | 0777);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_SYMLINKS);
	failure("Extracting a file through a symlink should fail here.");
	assertEqualInt(ARCHIVE_FAILED, archive_write_header(a, ae));
	archive_entry_free(ae);
	assert(0 == archive_write_finish_entry(a));

	/* Write an absolute symlink to /tmp. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "/tmp/libarchive_test-test_write_disk_secure-absolute_symlink");
	archive_entry_set_mode(ae, S_IFLNK | 0777);
	archive_entry_set_symlink(ae, "/tmp");
	archive_write_disk_set_options(a, 0);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));

	/* With security checks enabled, this should fail. */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "/tmp/libarchive_test-test_write_disk_secure-absolute_symlink/libarchive_test-test_write_disk_secure-absolute_symlink_path.tmp");
	archive_entry_set_mode(ae, S_IFREG | 0777);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_SYMLINKS);
	failure("Extracting a file through an absolute symlink should fail here.");
	assertEqualInt(ARCHIVE_FAILED, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertFileNotExists("/tmp/libarchive_test-test_write_disk_secure-absolute_symlink/libarchive_test-test_write_disk_secure-absolute_symlink_path.tmp");
	assert(0 == unlink("/tmp/libarchive_test-test_write_disk_secure-absolute_symlink"));
	unlink("/tmp/libarchive_test-test_write_disk_secure-absolute_symlink_path.tmp");

	/* Create another link. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir2");
	archive_entry_set_mode(ae, S_IFLNK | 0777);
	archive_entry_set_symlink(ae, "dir");
	archive_write_disk_set_options(a, 0);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));

	/*
	 * With symlink check and unlink option, it should remove
	 * the link and create the dir.
	 */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir2/filec");
	archive_entry_set_mode(ae, S_IFREG | 0777);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_SYMLINKS | ARCHIVE_EXTRACT_UNLINK);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assert(0 == archive_write_finish_entry(a));

	/* Create a nested symlink. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "dir/nested_link_to_dir");
	archive_entry_set_mode(ae, S_IFLNK | 0777);
	archive_entry_set_symlink(ae, "../dir");
	archive_write_disk_set_options(a, 0);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));

	/* But with security checks enabled, this should fail. */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "dir/nested_link_to_dir/filed");
	archive_entry_set_mode(ae, S_IFREG | 0777);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_SYMLINKS);
	failure("Extracting a file through a symlink should fail here.");
	assertEqualInt(ARCHIVE_FAILED, archive_write_header(a, ae));
	archive_entry_free(ae);
	assert(0 == archive_write_finish_entry(a));

	/*
	 * Without security checks, extracting a dir over a link to a
	 * dir should follow the link.
	 */
	/* Create a symlink to a dir. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir3");
	archive_entry_set_mode(ae, S_IFLNK | 0777);
	archive_entry_set_symlink(ae, "dir");
	archive_write_disk_set_options(a, 0);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));
	/* Extract a dir whose name matches the symlink. */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir3");
	archive_entry_set_mode(ae, S_IFDIR | 0777);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));
	/* Verify link was followed. */
	assertEqualInt(0, lstat("link_to_dir3", &st));
	assert(S_ISLNK(st.st_mode));
	archive_entry_free(ae);

	/*
	 * As above, but a broken link, so the link should get replaced.
	 */
	/* Create a symlink to a dir. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir4");
	archive_entry_set_mode(ae, S_IFLNK | 0777);
	archive_entry_set_symlink(ae, "nonexistent_dir");
	archive_write_disk_set_options(a, 0);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));
	/* Extract a dir whose name matches the symlink. */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir4");
	archive_entry_set_mode(ae, S_IFDIR | 0777);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));
	/* Verify link was replaced. */
	assertEqualInt(0, lstat("link_to_dir4", &st));
	assert(S_ISDIR(st.st_mode));
	archive_entry_free(ae);

	/*
	 * As above, but a link to a non-dir, so the link should get replaced.
	 */
	/* Create a regular file and a symlink to it */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "non_dir");
	archive_entry_set_mode(ae, S_IFREG | 0777);
	archive_write_disk_set_options(a, 0);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));
	/* Create symlink to the file. */
	archive_entry_copy_pathname(ae, "link_to_dir5");
	archive_entry_set_mode(ae, S_IFLNK | 0777);
	archive_entry_set_symlink(ae, "non_dir");
	archive_write_disk_set_options(a, 0);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));
	/* Extract a dir whose name matches the symlink. */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "link_to_dir5");
	archive_entry_set_mode(ae, S_IFDIR | 0777);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));
	/* Verify link was replaced. */
	assertEqualInt(0, lstat("link_to_dir5", &st));
	assert(S_ISDIR(st.st_mode));
	archive_entry_free(ae);

	/*
	 * Without security checks, we should be able to
	 * extract an absolute path.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "/tmp/libarchive_test-test_write_disk_secure-absolute_path.tmp");
	archive_entry_set_mode(ae, S_IFREG | 0777);
	assert(0 == archive_write_header(a, ae));
	assert(0 == archive_write_finish_entry(a));
	assertFileExists("/tmp/libarchive_test-test_write_disk_secure-absolute_path.tmp");
	assert(0 == unlink("/tmp/libarchive_test-test_write_disk_secure-absolute_path.tmp"));

	/* But with security checks enabled, this should fail. */
	assert(archive_entry_clear(ae) != NULL);
	archive_entry_copy_pathname(ae, "/tmp/libarchive_test-test_write_disk_secure-absolute_path.tmp");
	archive_entry_set_mode(ae, S_IFREG | 0777);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS);
	failure("Extracting an absolute path should fail here.");
	assertEqualInt(ARCHIVE_FAILED, archive_write_header(a, ae));
	archive_entry_free(ae);
	assert(0 == archive_write_finish_entry(a));
	assertFileNotExists("/tmp/libarchive_test-test_write_disk_secure-absolute_path.tmp");

	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Test the entries on disk. */
	assert(0 == lstat("dir", &st));
	failure("dir: st.st_mode=%o", st.st_mode);
	assert((st.st_mode & 0777) == 0755);

	assert(0 == lstat("link_to_dir", &st));
	failure("link_to_dir: st.st_mode=%o", st.st_mode);
	assert(S_ISLNK(st.st_mode));
#if defined(HAVE_SYMLINK) && defined(HAVE_LCHMOD) && \
    defined(S_IRUSR) && defined(S_IWUSR) && defined(S_IXUSR)
	/* Verify if we are able to lchmod() */
	if (symlink("dir", "testlink_to_dir") == 0) {
		if (lchmod("testlink_to_dir",
		    S_IRUSR | S_IWUSR | S_IXUSR) != 0) {
			switch (errno) {
				case ENOTSUP:
				case ENOSYS:
#if ENOTSUP != EOPNOTSUPP
				case EOPNOTSUPP:
#endif
					working_lchmod = 0;
					break;
				default:
					working_lchmod = 1;
			}
		} else
			working_lchmod = 1;
	} else
		working_lchmod = 0;

	if (working_lchmod) {
		failure("link_to_dir: st.st_mode=%o", st.st_mode);
		assert((st.st_mode & 07777) == 0755);
	}
#endif

	assert(0 == lstat("dir/filea", &st));
	failure("dir/filea: st.st_mode=%o", st.st_mode);
	assert((st.st_mode & 07777) == 0755);

	failure("dir/fileb: This file should not have been created");
	assert(0 != lstat("dir/fileb", &st));

	assert(0 == lstat("link_to_dir2", &st));
	failure("link_to_dir2 should have been re-created as a true dir");
	assert(S_ISDIR(st.st_mode));
	failure("link_to_dir2: Implicit dir creation should obey umask, but st.st_mode=%o", st.st_mode);
	assert((st.st_mode & 0777) == 0755);

	assert(0 == lstat("link_to_dir2/filec", &st));
	assert(S_ISREG(st.st_mode));
	failure("link_to_dir2/filec: st.st_mode=%o", st.st_mode);
	assert((st.st_mode & 07777) == 0755);

	failure("dir/filed: This file should not have been created");
	assert(0 != lstat("dir/filed", &st));
#endif
}
