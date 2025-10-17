/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
Execute the following to rebuild the data for this program:
   tail -n +35 test_read_format_isojoliet_rr.c | /bin/sh

rm -rf /tmp/iso
mkdir /tmp/iso
mkdir /tmp/iso/dir
file="long-joliet-file-name.textfile"
echo "hello" >/tmp/iso/$file
ln /tmp/iso/$file /tmp/iso/hardlink
(cd /tmp/iso; ln -s $file symlink)
if [ "$(uname -s)" = "Linux" ]; then # gnu coreutils touch doesn't have -h
TZ=utc touch -afm -t 197001020000.01 /tmp/iso/hardlink /tmp/iso/$file /tmp/iso/dir
TZ=utc touch -afm -t 197001030000.02 /tmp/iso/symlink
TZ=utc touch -afm -t 197001020000.01 /tmp/iso
else
TZ=utc touch -afhm -t 197001020000.01 /tmp/iso/hardlink /tmp/iso/$file /tmp/iso/dir
TZ=utc touch -afhm -t 197001030000.02 /tmp/iso/symlink
TZ=utc touch -afhm -t 197001020000.01 /tmp/iso
fi
F=test_read_format_iso_joliet_rockridge.iso.Z
mkhybrid -J -uid 1 -gid 2 /tmp/iso | compress > $F
uuencode $F $F > $F.uu
exit 1
 */

DEFINE_TEST(test_read_format_isojoliet_rr)
{
	const char *refname = "test_read_format_iso_joliet_rockridge.iso.Z";
	struct archive_entry *ae;
	struct archive *a;
	const void *p;
	size_t size;
	int64_t offset;

	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualInt(0, archive_read_support_filter_all(a));
	assertEqualInt(0, archive_read_support_format_all(a));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* First entry is '.' root directory. */
	assertEqualInt(0, archive_read_next_header(a, &ae));
	assertEqualString(".", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR, archive_entry_filetype(ae));
	assertEqualInt(2048, archive_entry_size(ae));
	assertEqualInt(86401, archive_entry_mtime(ae));
	assertEqualInt(0, archive_entry_mtime_nsec(ae));
	assertEqualInt(3, archive_entry_stat(ae)->st_nlink);
	assertEqualInt(1, archive_entry_uid(ae));
	assertEqualIntA(a, ARCHIVE_EOF,
	    archive_read_data_block(a, &p, &size, &offset));
	assertEqualInt((int)size, 0);
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* A directory. */
	assertEqualInt(0, archive_read_next_header(a, &ae));
	assertEqualString("dir", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR, archive_entry_filetype(ae));
	assertEqualInt(2048, archive_entry_size(ae));
	assertEqualInt(86401, archive_entry_mtime(ae));
	assertEqualInt(86401, archive_entry_atime(ae));
	assertEqualInt(2, archive_entry_stat(ae)->st_nlink);
	assertEqualInt(1, archive_entry_uid(ae));
	assertEqualInt(2, archive_entry_gid(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* A regular file with two names ("hardlink" gets returned
	 * first, so it's not marked as a hardlink). */
	assertEqualInt(0, archive_read_next_header(a, &ae));
	assertEqualString("hardlink",
	    archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assert(archive_entry_hardlink(ae) == NULL);
	assertEqualInt(6, archive_entry_size(ae));
	assertEqualInt(0, archive_read_data_block(a, &p, &size, &offset));
	assertEqualInt(6, (int)size);
	assertEqualInt(0, offset);
	assertEqualMem(p, "hello\n", 6);
	assertEqualInt(86401, archive_entry_mtime(ae));
	/* mkisofs records their access time. */
	/*assertEqualInt(86401, archive_entry_atime(ae));*/
	/* TODO: Actually, libarchive should be able to
	 * compute nlinks correctly even without RR
	 * extensions. See comments in libarchive source. */
	assertEqualInt(2, archive_entry_nlink(ae));
	assertEqualInt(1, archive_entry_uid(ae));
	assertEqualInt(2, archive_entry_gid(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Second name for the same regular file (this happens to be
	 * returned second, so does get marked as a hardlink). */
	assertEqualInt(0, archive_read_next_header(a, &ae));
	assertEqualString("long-joliet-file-name.textfile",
	    archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assertEqualString("hardlink",
	    archive_entry_hardlink(ae));
	assert(!archive_entry_size_is_set(ae));
	assertEqualInt(86401, archive_entry_mtime(ae));
	assertEqualInt(86401, archive_entry_atime(ae));
	/* TODO: See above. */
	assertEqualInt(2, archive_entry_nlink(ae));
	assertEqualInt(1, archive_entry_uid(ae));
	assertEqualInt(2, archive_entry_gid(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* A symlink to the regular file. */
	assertEqualInt(0, archive_read_next_header(a, &ae));
	assertEqualString("symlink", archive_entry_pathname(ae));
	assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
	assertEqualString("long-joliet-file-name.textfile",
	    archive_entry_symlink(ae));
	assertEqualInt(0, archive_entry_size(ae));
	assertEqualInt(172802, archive_entry_mtime(ae));
	assertEqualInt(172802, archive_entry_atime(ae));
	assertEqualInt(1, archive_entry_nlink(ae));
	assertEqualInt(1, archive_entry_uid(ae));
	assertEqualInt(2, archive_entry_gid(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* End of archive. */
	assertEqualInt(ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_COMPRESS);
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_ISO9660_ROCKRIDGE);

	/* Close the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

