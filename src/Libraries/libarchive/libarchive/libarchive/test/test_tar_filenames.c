/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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

/*
 * Exercise various lengths of filenames in tar archives,
 * especially around the magic sizes where ustar breaks
 * filenames into prefix/suffix.
 */

static void
test_filename(const char *prefix, int dlen, int flen)
{
	char buff[8192];
	char filename[400];
	char dirname[400];
	struct archive_entry *ae;
	struct archive *a;
	size_t used;
	char *p;
	int i;

	p = filename;
	if (prefix) {
		strcpy(filename, prefix);
		p += strlen(p);
	}
	if (dlen > 0) {
		for (i = 0; i < dlen; i++)
			*p++ = 'a';
		*p++ = '/';
	}
	for (i = 0; i < flen; i++)
		*p++ = 'b';
	*p = '\0';

	strcpy(dirname, filename);

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertA(0 == archive_write_set_format_pax_restricted(a));
	assertA(0 == archive_write_add_filter_none(a));
	assertA(0 == archive_write_set_bytes_per_block(a,0));
	assertA(0 == archive_write_open_memory(a, buff, sizeof(buff), &used));

	/*
	 * Write a file to it.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, filename);
	archive_entry_set_mode(ae, S_IFREG | 0755);
	failure("Pathname %d/%d", dlen, flen);
	assertA(0 == archive_write_header(a, ae));
	archive_entry_free(ae);

	/*
	 * Write a dir to it (without trailing '/').
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, dirname);
	archive_entry_set_mode(ae, S_IFDIR | 0755);
	failure("Dirname %d/%d", dlen, flen);
	assertA(0 == archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Tar adds a '/' to directory names. */
	strcat(dirname, "/");

	/*
	 * Write a dir to it (with trailing '/').
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, dirname);
	archive_entry_set_mode(ae, S_IFDIR | 0755);
	failure("Dirname %d/%d", dlen, flen);
	assertA(0 == archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Close out the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/*
	 * Now, read the data back.
	 */
	assert((a = archive_read_new()) != NULL);
	assertA(0 == archive_read_support_format_all(a));
	assertA(0 == archive_read_support_filter_all(a));
	assertA(0 == archive_read_open_memory(a, buff, used));

	/* Read the file and check the filename. */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualString(filename, archive_entry_pathname(ae));
	assertEqualInt((S_IFREG | 0755), archive_entry_mode(ae));

	/*
	 * Read the two dirs and check the names.
	 *
	 * Both dirs should read back with the same name, since
	 * tar should add a trailing '/' to any dir that doesn't
	 * already have one.  We only report the first such failure
	 * here.
	 */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualString(dirname, archive_entry_pathname(ae));
	assert((S_IFDIR | 0755) == archive_entry_mode(ae));

	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualString(dirname, archive_entry_pathname(ae));
	assert((S_IFDIR | 0755) == archive_entry_mode(ae));

	/* Verify the end of the archive. */
	assert(1 == archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_tar_filenames)
{
	int dlen, flen;

	/* Repeat the following for a variety of dir/file lengths. */
	for (dlen = 45; dlen < 55; dlen++) {
		for (flen = 45; flen < 55; flen++) {
			test_filename(NULL, dlen, flen);
			test_filename("/", dlen, flen);
		}
	}

	for (dlen = 0; dlen < 140; dlen += 10) {
		for (flen = 98; flen < 102; flen++) {
			test_filename(NULL, dlen, flen);
			test_filename("/", dlen, flen);
		}
	}

	for (dlen = 140; dlen < 160; dlen++) {
		for (flen = 95; flen < 105; flen++) {
			test_filename(NULL, dlen, flen);
			test_filename("/", dlen, flen);
		}
	}
}
