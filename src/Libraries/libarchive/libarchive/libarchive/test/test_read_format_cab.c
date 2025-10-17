/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
   tail -n +44 test_read_format_cab.c | /bin/sh
And following works are:
1. Move /tmp/cab/cab.zip to Windows PC
2. Extract cab.zip
3. Open command prompt and change current directory where you extracted cab.zip
4. Execute cab.bat
5. Then you will see that there is a cabinet file, test.cab
6. Move test.cab to posix platform
7. Extract test.cab with this version of bsdtar
8. Execute the following command to make uuencoded files.
 uuencode test_read_format_cab_1.cab test_read_format_cab_1.cab > test_read_format_cab_1.cab.uu
 uuencode test_read_format_cab_2.cab test_read_format_cab_2.cab > test_read_format_cab_2.cab.uu
 uuencode test_read_format_cab_3.cab test_read_format_cab_3.cab > test_read_format_cab_3.cab.uu

#!/bin/sh
#
# How to make test data.
#
# Temporary directory.
base=/tmp/cab
# Owner id
owner=1001
# Group id
group=1001
#
# Make contents of a cabinet file.
#
rm -rf ${base}
mkdir ${base}
mkdir ${base}/dir1
mkdir ${base}/dir2
#
touch ${base}/empty
cat > ${base}/dir1/file1 << END
                          file 1 contents
hello
hello
hello
END
#
cat > ${base}/dir2/file2 << END
                          file 2 contents
hello
hello
hello
hello
hello
hello
END
#
dd if=/dev/zero of=${base}/zero bs=1 count=33000 > /dev/null 2>&1
#
cab1=test_read_format_cab_1.cab
cab2=test_read_format_cab_2.cab
cab3=test_read_format_cab_3.cab
#
#
cat > ${base}/mkcab1 << END
.Set Compress=OFF
.Set DiskDirectory1=.
.Set InfDate=1980-01-02
.Set InfTime=00:00:00
.Set CabinetName1=${cab1}
empty
.Set DestinationDir=dir1
dir1/file1
.Set DestinationDir=dir2
dir2/file2
END
#
cat > ${base}/mkcab2 << END
.Set CompressionType=MSZIP
.Set DiskDirectory1=.
.Set InfDate=1980-01-02
.Set InfTime=00:00:00
.Set CabinetName1=${cab2}
empty
zero
.Set DestinationDir=dir1
dir1/file1
.Set DestinationDir=dir2
dir2/file2
END
#
cat > ${base}/mkcab3 << END
.Set CompressionType=LZX
.Set DiskDirectory1=.
.Set InfDate=1980-01-02
.Set InfTime=00:00:00
.Set CabinetName1=${cab3}
empty
zero
.Set DestinationDir=dir1
dir1/file1
.Set DestinationDir=dir2
dir2/file2
END
#
cat > ${base}/mkcab4 << END
.Set CompressionType=MSZIP
.Set DiskDirectory1=.
.Set CabinetName1=test.cab
${cab1}
${cab2}
${cab3}
END
#
cat > ${base}/cab.bat << END
makecab.exe /F mkcab1
makecab.exe /F mkcab2
makecab.exe /F mkcab3
makecab.exe /F mkcab4
del setup.inf setup.rpt
del empty zero dir1\file1 dir2\file2 mkcab1 mkcab2 mkcab3 mkcab4
del ${cab1} ${cab2} ${cab3}
rmdir dir1 dir2
END
#
f=cab.zip
(cd ${base}; zip -q -c $f empty zero dir1/file1 dir2/file2 mkcab1 mkcab2 mkcab3 mkcab4 cab.bat)
#
exit 1
*/

static const char file1[] = {
"                          file 1 contents\n"
"hello\n"
"hello\n"
"hello\n"
};
#define file1_size (sizeof(file1)-1)
static const char file2[] = {
"                          file 2 contents\n"
"hello\n"
"hello\n"
"hello\n"
"hello\n"
"hello\n"
"hello\n"
};
#define file2_size (sizeof(file2)-1)

enum comp_type {
	STORE = 0,
	MSZIP,
	LZX
};
static void
verify(const char *refname, enum comp_type comp)
{
	struct archive_entry *ae;
	struct archive *a;
	char buff[128];
	char zero[128];
	size_t s;

	memset(zero, 0, sizeof(zero));
	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular empty. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt((AE_IFREG | 0666), archive_entry_mode(ae));
	assertEqualString("empty", archive_entry_pathname(ae));
	assertEqualInt(0, archive_entry_uid(ae));
	assertEqualInt(0, archive_entry_gid(ae));
	assertEqualInt(0, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	if (comp != STORE) {
		/* Verify regular zero.
		 * Maximum CFDATA size is 32768, so we need over 32768 bytes
		 * file to check if we properly handle multiple CFDATA.
		 */
		assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
		assertEqualInt((AE_IFREG | 0666), archive_entry_mode(ae));
		assertEqualString("zero", archive_entry_pathname(ae));
		assertEqualInt(0, archive_entry_uid(ae));
		assertEqualInt(0, archive_entry_gid(ae));
		assertEqualInt(archive_entry_is_encrypted(ae), 0);
		assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
		assertEqualInt(33000, archive_entry_size(ae));
		for (s = 0; s + sizeof(buff) < 33000; s+= sizeof(buff)) {
			ssize_t rsize = archive_read_data(a, buff, sizeof(buff));
			if (comp == MSZIP && rsize == ARCHIVE_FATAL && archive_zlib_version() == NULL) {
				skipping("Skipping CAB format(MSZIP) check: %s",
				    archive_error_string(a));
				goto finish;
			}
			assertEqualInt(sizeof(buff), rsize);
			assertEqualMem(buff, zero, sizeof(buff));
		}
		assertEqualInt(33000 - s, archive_read_data(a, buff, 33000 - s));
		assertEqualMem(buff, zero, 33000 - s);
	}

	/* Verify regular file1. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt((AE_IFREG | 0666), archive_entry_mode(ae));
	assertEqualString("dir1/file1", archive_entry_pathname(ae));
	assertEqualInt(0, archive_entry_uid(ae));
	assertEqualInt(0, archive_entry_gid(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualInt(file1_size, archive_entry_size(ae));
	assertEqualInt(file1_size, archive_read_data(a, buff, file1_size));
	assertEqualMem(buff, file1, file1_size);

	/* Verify regular file2. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt((AE_IFREG | 0666), archive_entry_mode(ae));
	assertEqualString("dir2/file2", archive_entry_pathname(ae));
	assertEqualInt(0, archive_entry_uid(ae));
	assertEqualInt(0, archive_entry_gid(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualInt(file2_size, archive_entry_size(ae));
	assertEqualInt(file2_size, archive_read_data(a, buff, file2_size));
	assertEqualMem(buff, file2, file2_size);

	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	if (comp != STORE) {
		assertEqualInt(4, archive_file_count(a));
	} else {
		assertEqualInt(3, archive_file_count(a));
	}

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CAB, archive_format(a));

	/* Close the archive. */
finish:
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

/*
 * Skip beginning files and Read the last file.
 */
static void
verify2(const char *refname, enum comp_type comp)
{
	struct archive_entry *ae;
	struct archive *a;
	char buff[128];
	char zero[128];

	if (comp == MSZIP && archive_zlib_version() == NULL) {
		skipping("Skipping CAB format(MSZIP) check for %s",
		  refname);
		return;
	}
	memset(zero, 0, sizeof(zero));
	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular empty. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	if (comp != STORE) {
		assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
		assertEqualInt(archive_entry_is_encrypted(ae), 0);
		assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	}
	/* Verify regular file1. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Verify regular file2. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt((AE_IFREG | 0666), archive_entry_mode(ae));
	assertEqualString("dir2/file2", archive_entry_pathname(ae));
	assertEqualInt(0, archive_entry_uid(ae));
	assertEqualInt(0, archive_entry_gid(ae));
	assertEqualInt(file2_size, archive_entry_size(ae));
	assertEqualInt(file2_size, archive_read_data(a, buff, file2_size));
	assertEqualMem(buff, file2, file2_size);

	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	if (comp != STORE) {
		assertEqualInt(4, archive_file_count(a));
	} else {
		assertEqualInt(3, archive_file_count(a));
	}

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CAB, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

/*
 * Skip all file like 'bsdtar tvf foo.cab'.
 */
static void
verify3(const char *refname, enum comp_type comp)
{
	struct archive_entry *ae;
	struct archive *a;
	char zero[128];

	memset(zero, 0, sizeof(zero));
	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular empty. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	if (comp != STORE) {
		assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
		assertEqualInt(archive_entry_is_encrypted(ae), 0);
		assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	}
	/* Verify regular file1. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Verify regular file2. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	if (comp != STORE) {
		assertEqualInt(4, archive_file_count(a));
	} else {
		assertEqualInt(3, archive_file_count(a));
	}

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CAB, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cab)
{
	/* Verify Cabinet file in no compression. */
	verify("test_read_format_cab_1.cab", STORE);
	verify2("test_read_format_cab_1.cab", STORE);
	verify3("test_read_format_cab_1.cab", STORE);
	/* Verify Cabinet file in MSZIP. */
	verify("test_read_format_cab_2.cab", MSZIP);
	verify2("test_read_format_cab_2.cab", MSZIP);
	verify3("test_read_format_cab_2.cab", MSZIP);
	/* Verify Cabinet file in LZX. */
	verify("test_read_format_cab_3.cab", LZX);
	verify2("test_read_format_cab_3.cab", LZX);
	verify3("test_read_format_cab_3.cab", LZX);
}


