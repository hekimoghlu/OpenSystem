/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

DEFINE_TEST(test_write_disk_failures)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	skipping("archive_write_disk interface");
#else
	struct archive_entry *ae;
	struct archive *a;
	int fd;

	/* Force the umask to something predictable. */
	assertUmask(022);

	/* A directory that we can't write to. */
	assertMakeDir("dir", 0555);

	/* Can we? */
	fd = open("dir/testfile", O_WRONLY | O_CREAT | O_BINARY, 0777);
	if (fd >= 0) {
	  /* Apparently, we can, so the test below won't work. */
	  close(fd);
	  skipping("Can't test writing to non-writable directory");
	  return;
	}

	/* Try to extract a regular file into the directory above. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "dir/file");
	archive_entry_set_mode(ae, S_IFREG | 0755);
	archive_entry_set_size(ae, 8);
	assert((a = archive_write_disk_new()) != NULL);
        archive_write_disk_set_options(a, ARCHIVE_EXTRACT_TIME);
	archive_entry_set_mtime(ae, 123456789, 0);
	assertEqualIntA(a, ARCHIVE_FAILED, archive_write_header(a, ae));
	assertEqualIntA(a, 0, archive_write_finish_entry(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));
	archive_entry_free(ae);
#endif
}
