/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
Execute the following command to rebuild the data for this program:
   tail -n +32 test_read_format_isorr_ce.c | /bin/sh

dirname=/tmp/iso
#
rm -rf $dirname
mkdir $dirname
#
num=0
file=""
while [ $num -lt 150 ]
do
  num=$((num+1))
  file="a$file"
done
#
num=0
while [ $num -lt 3 ]
do
  num=$((num+1))
  file="a$file"
  echo "hello $((num+150))" > $dirname/$file
  dd if=/dev/zero count=1 bs=4080 >> $dirname/$file
  (cd $dirname; ln -s $file sym$num)
done
#
mkdir $dirname/dir
#
time1="197001020000.01"
time2="197001030000.02"
TZ=utc touch -afhm -t $time1 $dirname/dir $dirname/aaaa*
TZ=utc touch -afhm -t $time2 $dirname/sym*
TZ=utc touch -afhm -t $time1 $dirname
#
F=test_read_format_iso_rockridge_ce.iso.Z
mkisofs -R -uid 1 -gid 2 $dirname | compress > $F
uuencode $F $F > $F.uu
rm -rf $dirname
exit 1
 */

/*
 * Test reading SUSP "CE" extension is works fine.
 */

static void
mkpath(char *p, int len)
{
	int i;

	for (i = 0; i < len; i++)
		p[i] = 'a';
	p[len] = '\0';
}

DEFINE_TEST(test_read_format_isorr_ce)
{
	const char *refname = "test_read_format_iso_rockridge_ce.iso.Z";
	char path1[160];
	char path2[160];
	char path3[160];
	struct archive_entry *ae;
	struct archive *a;
	const void *p;
	size_t size;
	int64_t offset;
	int i;

	mkpath(path1, 151);
	mkpath(path2, 152);
	mkpath(path3, 153);
	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualInt(0, archive_read_support_filter_all(a));
	assertEqualInt(0, archive_read_support_format_all(a));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Retrieve each of the 8 files on the ISO image and
	 * verify that each one is what we expect. */
	for (i = 0; i < 8; ++i) {
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
		} else if (strcmp(path1, archive_entry_pathname(ae)) == 0) {
			/* A regular file. */
			assertEqualString(path1, archive_entry_pathname(ae));
			assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
			assertEqualInt(4090, archive_entry_size(ae));
			assertEqualInt(0,
			    archive_read_data_block(a, &p, &size, &offset));
			assertEqualInt(0, offset);
			assertEqualMem(p, "hello 151\n", 10);
			assertEqualInt(86401, archive_entry_mtime(ae));
			assertEqualInt(86401, archive_entry_atime(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp(path2, archive_entry_pathname(ae)) == 0) {
			/* A regular file. */
			assertEqualString(path2, archive_entry_pathname(ae));
			assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
			assertEqualInt(4090, archive_entry_size(ae));
			assertEqualInt(0,
			    archive_read_data_block(a, &p, &size, &offset));
			assertEqualInt(0, offset);
			assertEqualMem(p, "hello 152\n", 10);
			assertEqualInt(86401, archive_entry_mtime(ae));
			assertEqualInt(86401, archive_entry_atime(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp(path3, archive_entry_pathname(ae)) == 0) {
			/* A regular file. */
			assertEqualString(path3, archive_entry_pathname(ae));
			assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
			assertEqualInt(4090, archive_entry_size(ae));
			assertEqualInt(0,
			    archive_read_data_block(a, &p, &size, &offset));
			assertEqualInt(0, offset);
			assertEqualMem(p, "hello 153\n", 10);
			assertEqualInt(86401, archive_entry_mtime(ae));
			assertEqualInt(86401, archive_entry_atime(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("sym1", archive_entry_pathname(ae)) == 0) {
			/* A symlink to the regular file. */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString(path1, archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(172802, archive_entry_mtime(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("sym2", archive_entry_pathname(ae)) == 0) {
			/* A symlink to the regular file. */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString(path2, archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(172802, archive_entry_mtime(ae));
			assertEqualInt(1, archive_entry_stat(ae)->st_nlink);
			assertEqualInt(1, archive_entry_uid(ae));
			assertEqualInt(2, archive_entry_gid(ae));
		} else if (strcmp("sym3", archive_entry_pathname(ae)) == 0) {
			/* A symlink to the regular file. */
			assertEqualInt(AE_IFLNK, archive_entry_filetype(ae));
			assertEqualString(path3, archive_entry_symlink(ae));
			assertEqualInt(0, archive_entry_size(ae));
			assertEqualInt(172802, archive_entry_mtime(ae));
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


