/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
execute the following to rebuild the data for this program:
   tail -n +33 test_read_format_cpio_afio.c | /bin/sh

# How to make a sample data.
echo "0123456789abcdef" > file1
echo "0123456789abcdef" > file2
# make afio use a large ASCII header
sudo chown 65536 file2
find . -name "file[12]" | afio -o sample
od -c sample | sed -E -e "s/^0[0-9]+//;s/^  //;s/( +)([^ ]{1,2})/'\2',/g;s/'\\0'/0/g;/^[*]/d" > test_read_format_cpio_afio.sample.txt
rm -f file1 file2 sample
exit1
*/

static unsigned char archive[] = {
'0','7','0','7','0','7','0','0','0','1','4','3','1','2','5','3',
'2','1','1','0','0','6','4','4','0','0','1','7','5','1','0','0',
'1','7','5','1','0','0','0','0','0','1','0','0','0','0','0','0',
'1','1','3','3','2','2','4','5','0','2','0','0','0','0','0','0',
'6','0','0','0','0','0','0','0','0','0','2','1','f','i','l','e',
'1',0,'0','1','2','3','4','5','6','7','8','9','a','b','c','d',
'e','f','\n','0','7','0','7','2','7','0','0','0','0','0','0','6',
'3','0','0','0','0','0','0','0','0','0','0','0','D','A','A','E',
'6','m','1','0','0','6','4','4','0','0','0','1','0','0','0','0',
'0','0','0','0','0','3','E','9','0','0','0','0','0','0','0','1',
'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0',
'4','B','6','9','4','A','1','0','n','0','0','0','6','0','0','0',
'0','0','0','0','0','s','0','0','0','0','0','0','0','0','0','0',
'0','0','0','0','1','1',':','f','i','l','e','2',0,'0','1','2',
'3','4','5','6','7','8','9','a','b','c','d','e','f','\n','0','7',
'0','7','0','7','0','0','0','0','0','0','0','0','0','0','0','0',
'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0',
'0','0','0','0','0','0','0','1','0','0','0','0','0','0','0','0',
'0','0','0','0','0','0','0','0','0','0','0','0','0','1','3','0',
'0','0','0','0','0','1','1','2','7','3','T','R','A','I','L','E',
'R','!','!','!',0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

DEFINE_TEST(test_read_format_cpio_afio)
{
	unsigned char *p;
	size_t size;
	struct archive_entry *ae;
	struct archive *a;

	/* The default block size of afio is 5120. we simulate it */
	size = (sizeof(archive) + 5120 -1 / 5120) * 5120;
	assert((p = malloc(size)) != NULL);
	if (p == NULL)
		return;
	memset(p, 0, size);
	memcpy(p, archive, sizeof(archive));
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, p, size));
	/*
	 * First entry is odc format.
	 */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(17, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertA(archive_filter_code(a, 0) == ARCHIVE_FILTER_NONE);
	assertA(archive_format(a) == ARCHIVE_FORMAT_CPIO_POSIX);
	/*
	 * Second entry is afio large ASCII format.
	 */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(17, archive_entry_size(ae));
	assertEqualInt(65536, archive_entry_uid(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertA(archive_filter_code(a, 0) == ARCHIVE_FILTER_NONE);
	assertA(archive_format(a) == ARCHIVE_FORMAT_CPIO_AFIO_LARGE);
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	free(p);
}

// From OSS Fuzz Issue 70019:
static unsigned char archive2[] = "070727bbbBbbbBabbbbbbcbcbbbbbbm726f777f777ffffffff518402ffffbbbabDDDDDDDDD7c7Ddd7DDDDnDDDdDDDB7777s77777777777C7727:";

DEFINE_TEST(test_read_format_cpio_afio_broken)
{
	struct archive *a;
	struct archive_entry *ae;

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, archive2, sizeof(archive2)));
	assertEqualIntA(a, ARCHIVE_FATAL, archive_read_next_header(a, &ae));
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_NONE);
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_CPIO_AFIO_LARGE);
	archive_read_free(a);
}
