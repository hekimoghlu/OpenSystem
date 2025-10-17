/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
PLEASE use old cdrtools; mkisofs version is 2.01.
This version mkisofs made wrong "SL" System Use Entry of RRIP.

Execute the following command to rebuild the data for this program:
   tail -n +34 test_read_format_isorr_bz2.c | /bin/sh

rm -rf /tmp/iso
mkdir /tmp/iso
mkdir /tmp/iso/dir
echo "hello" >/tmp/iso/file
dd if=/dev/zero count=1 bs=12345678 >>/tmp/iso/file
ln /tmp/iso/file /tmp/iso/hardlink
(cd /tmp/iso; ln -s file symlink)
(cd /tmp/iso; ln -s /tmp/ symlink2)
(cd /tmp/iso; ln -s /tmp/../ symlink3)
(cd /tmp/iso; ln -s .././../tmp/ symlink4)
(cd /tmp/iso; ln -s .///file symlink5)
(cd /tmp/iso; ln -s /tmp//../ symlink6)
TZ=utc touch -afhm -t 197001020000.01 /tmp/iso /tmp/iso/file /tmp/iso/dir
TZ=utc touch -afhm -t 197001030000.02 /tmp/iso/symlink /tmp/iso/symlink5
F=test_read_format_iso_rockridge.iso.Z
mkhybrid -R -uid 1 -gid 2 /tmp/iso | compress > $F
uuencode $F $F > $F.uu
exit 1
 */

DEFINE_TEST(test_read_format_isorr_bz2)
{
	const char *refname = "test_read_format_iso_rockridge.iso.Z";
	struct archive_entry *ae;
	struct archive *a;
	const void *p;
	size_t size;
	int64_t offset;
	int i;

	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualInt(0, archive_read_support_filter_all(a));
	assertEqualInt(0, archive_read_support_format_all(a));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Retrieve each of the 8 files on the ISO image and
	 * verify that each one is what we expect. */
	for (i = 0; i < 10; ++i) {
		assertEqualInt(0, archive_read_next_header(a, &ae));
		
		assertEqualInt(archive_entry_is_encrypted(ae), 0);
		assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

		if (strcmp(".", archive_entry_pathname(ae)) == 0) {
			/* '.' root directory. */
			assertEqualInt(AE_IFDIR, archive_entry_filetype(ae));
			assertEqualInt(2048, archive_entry_size(ae));
			/* Now, we read timestamp recorded by RRIP "TF". */ 
			assertEqualInt(86401, archive_entry_mtime(ae));
			assertEqualInt(0, archive_entry_mtime_nsec(ae));
			/* Now, we read links recorded by RRIP "PX". */ 
			assertEqualInt(3, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualIntA(a, ARCHIVE_EOF,
			    archive_read_data_block(a, &p, &size, &offset));
			assertEqualInt((int)size, 0);
		} else if (strcmp("dir", archive_entry_pathname(ae)) == 0) {
			/* A directory. */
			assertEqualString("dir", archive_entry_pathname(ae));
			assertEqualInt(AE_IFDIR, archive_entry_filetype(ae));
			assertEqualInt(2048, archive_entry_size(ae));
			assertEqualInt(86401, archive_entry_mtime(ae));
			assertEqualInt(86401, archive_entry_atime(ae));
			assertEqualInt(2, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("file", archive_entry_pathname(ae)) == 0) {
			/* A regular file. */
			assertEqualString("file", archive_entry_pathname(ae));
			assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
			assertEqualInt(12345684, archive_entry_size(ae));
			assertEqualInt(0,
			    archive_read_data_block(a, &p, &size, &offset));
			assertEqualInt(0, offset);
			assertEqualMem(p, "hello\n", 6);
			assertEqualInt(86401, archive_entry_mtime(ae));
			assertEqualInt(86401, archive_entry_atime(ae));
			assertEqualInt(2, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("hardlink", archive_entry_pathname(ae)) == 0) {
			/* A hardlink to the regular file. */
			/* Note: If "hardlink" gets returned before "file",
			 * then "hardlink" will get returned as a regular file
			 * and "file" will get returned as the hardlink.
			 * This test should tolerate that, since it's a
			 * perfectly permissible thing for libarchive to do. */
			assertEqualString("hardlink", archive_entry_pathname(ae));
			assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
			assertEqualString("file", archive_entry_hardlink(ae));
			assertEqualInt(0, archive_entry_size_is_set(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(86401, archive_entry_mtime(ae));
			assertEqualInt(86401, archive_entry_atime(ae));
			assertEqualInt(2, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("symlink", archive_entry_pathname(ae)) == 0) {
			/* A symlink to the regular file. */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString("file", archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(172802, archive_entry_mtime(ae));
			assertEqualInt(172802, archive_entry_atime(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("symlink2", archive_entry_pathname(ae)) == 0) {
			/* A symlink to /tmp (an absolute path) */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString("/tmp", archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("symlink3", archive_entry_pathname(ae)) == 0) {
			/* A symlink to /tmp/.. (with a ".." component) */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString("/tmp/..", archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("symlink4", archive_entry_pathname(ae)) == 0) {
			/* A symlink to a path with ".." and "." components */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString(".././../tmp",
			    archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("symlink5", archive_entry_pathname(ae)) == 0) {
			/* A symlink to the regular file with "/" components. */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString(".///file", archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(172802, archive_entry_mtime(ae));
			assertEqualInt(172802, archive_entry_atime(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("symlink6", archive_entry_pathname(ae)) == 0) {
			/* A symlink to /tmp//..
			 * (with "/" and ".." components) */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString("/tmp//..", archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else {
			failure("Saw a file that shouldn't have been there");
			assertEqualString(archive_entry_pathname(ae), "");
		}
	}

	/* End of archive. */
	assertEqualInt(ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_COMPRESS);
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_ISO9660_ROCKRIDGE);

	/* Close the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}


