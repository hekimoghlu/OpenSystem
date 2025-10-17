/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
 * Github Issue #745 describes a bug in the sandboxing code that
 * allows one to use a symlink to edit the permissions on a file or
 * directory outside of the sandbox.
 */

DEFINE_TEST(test_write_disk_secure745)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	skipping("archive_write_disk security checks not supported on Windows");
#else
	struct archive *a;
	struct archive_entry *ae;

	/* Start with a known umask. */
	assertUmask(UMASK);

	/* Create an archive_write_disk object. */
	assert((a = archive_write_disk_new()) != NULL);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_SYMLINKS);

	/* The target dir:  The one we're going to try to change permission on */
	assertMakeDir("target", 0700);

	/* The sandbox dir we're going to run inside of. */
	assertMakeDir("sandbox", 0700);
	assertChdir("sandbox");

	/* Create a symlink pointing to the target directory */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "sym");
	archive_entry_set_mode(ae, AE_IFLNK | 0777);
	archive_entry_copy_symlink(ae, "../target");
	assert(0 == archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Try to alter the target dir through the symlink; this should fail. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "sym");
	archive_entry_set_mode(ae, S_IFDIR | 0777);
	assert(0 == archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Permission of target dir should not have changed. */
	assertFileMode("../target", 0700);

	assert(0 == archive_write_close(a));
	archive_write_free(a);
#endif
}
