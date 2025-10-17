/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
 * Github Issue #746 describes a problem in which hardlink targets are
 * not adequately checked and can be used to modify entries outside of
 * the sandbox.
 */

/*
 * Verify that ARCHIVE_EXTRACT_SECURE_NODOTDOT disallows '..' in hardlink
 * targets.
 */
DEFINE_TEST(test_write_disk_secure746a)
{
	struct archive *a;
	struct archive_entry *ae;

	/* Start with a known umask. */
	assertUmask(UMASK);

	/* The target directory we're going to try to affect. */
	assertMakeDir("target", 0700);
	assertMakeFile("target/foo", 0700, "unmodified");

	/* The sandbox dir we're going to work within. */
	assertMakeDir("sandbox", 0700);
	assertChdir("sandbox");

	/* Create an archive_write_disk object. */
	assert((a = archive_write_disk_new()) != NULL);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_NODOTDOT);

	/* Attempt to hardlink to the target directory. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "bar");
	archive_entry_set_mode(ae, AE_IFREG | 0777);
	archive_entry_set_size(ae, 8);
	archive_entry_copy_hardlink(ae, "../target/foo");
	assertEqualInt(ARCHIVE_FAILED, archive_write_header(a, ae));
	assertEqualInt(ARCHIVE_FATAL, archive_write_data(a, "modified", 8));
	archive_entry_free(ae);

	/* Verify that target file contents are unchanged. */
	assertTextFileContents("unmodified", "../target/foo");

	assertEqualIntA(a, ARCHIVE_FATAL, archive_write_close(a));
	archive_write_free(a);
}

/*
 * Verify that ARCHIVE_EXTRACT_SECURE_NOSYMLINK disallows symlinks in hardlink
 * targets.
 */
DEFINE_TEST(test_write_disk_secure746b)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	skipping("archive_write_disk security checks not supported on Windows");
#else
	struct archive *a;
	struct archive_entry *ae;

	/* Start with a known umask. */
	assertUmask(UMASK);

	/* The target directory we're going to try to affect. */
	assertMakeDir("target", 0700);
	assertMakeFile("target/foo", 0700, "unmodified");

	/* The sandbox dir we're going to work within. */
	assertMakeDir("sandbox", 0700);
	assertChdir("sandbox");

	/* Create an archive_write_disk object. */
	assert((a = archive_write_disk_new()) != NULL);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_SYMLINKS);

	/* Create a symlink to the target directory. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "symlink");
	archive_entry_set_mode(ae, AE_IFLNK | 0777);
	archive_entry_copy_symlink(ae, "../target");
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Attempt to hardlink to the target directory via the symlink. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "bar");
	archive_entry_set_mode(ae, AE_IFREG | 0777);
	archive_entry_set_size(ae, 8);
	archive_entry_copy_hardlink(ae, "symlink/foo");
	assertEqualIntA(a, ARCHIVE_FAILED, archive_write_header(a, ae));
	assertEqualIntA(a, ARCHIVE_FATAL, archive_write_data(a, "modified", 8));
	archive_entry_free(ae);

	/* Verify that target file contents are unchanged. */
	assertTextFileContents("unmodified", "../target/foo");

	assertEqualIntA(a, ARCHIVE_FATAL, archive_write_close(a));
	archive_write_free(a);
#endif
}
