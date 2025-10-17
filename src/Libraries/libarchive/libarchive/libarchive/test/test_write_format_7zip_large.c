/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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

#define LARGE_SIZE	(1*1024*1024)
static void
test_large(const char *compression_type)
{
	struct archive_entry *ae;
	struct archive *a;
	size_t used;
	size_t buffsize = LARGE_SIZE + 1024 * 256;
	size_t datasize = LARGE_SIZE;
	char *buff, *filedata, *filedata2;

	assert((buff = malloc(buffsize)) != NULL);
	assert((filedata = malloc(datasize)) != NULL);
	assert((filedata2 = malloc(datasize)) != NULL);

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	if (a == NULL || buff == NULL || filedata == NULL || filedata2 == NULL) {
		archive_write_free(a);
		free(buff);
		free(filedata);
		free(filedata2);
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_7zip(a));
	if (compression_type != NULL &&
	    ARCHIVE_OK != archive_write_set_format_option(a, "7zip",
	    "compression", compression_type)) {
		skipping("%s writing not fully supported on this platform",
		   compression_type);
		assertEqualInt(ARCHIVE_OK, archive_write_free(a));
		free(buff);
		free(filedata);
		free(filedata2);
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_write_add_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, buffsize, &used));

	/*
	 * Write a large file to it.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_mtime(ae, 1, 100);
	assertEqualInt(1, archive_entry_mtime(ae));
	assertEqualInt(100, archive_entry_mtime_nsec(ae));
	archive_entry_copy_pathname(ae, "file");
	assertEqualString("file", archive_entry_pathname(ae));
	archive_entry_set_mode(ae, AE_IFREG | 0755);
	assertEqualInt((AE_IFREG | 0755), archive_entry_mode(ae));
	archive_entry_set_size(ae, datasize);

	assertEqualInt(0, archive_write_header(a, ae));
	archive_entry_free(ae);
	if (strcmp(compression_type, "ppmd") == 0) {
		/* NOTE: PPMd cannot handle random data correctly.*/
		memset(filedata, 'a', datasize);
	} else {
		fill_with_pseudorandom_data(filedata, datasize);
	}
	assertEqualInt(datasize, archive_write_data(a, filedata, datasize));

	/* Close out the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Verify the initial header. */
	assertEqualMem(buff, "\x37\x7a\xbc\xaf\x27\x1c\x00\x03", 8);

	/*
	 * Now, read the data back.
	 */
	/* With the test memory reader -- seeking mode. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory_seek(a, buff, used, 7));

	/*
	 * Read and verify a large file.
	 */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(1, archive_entry_mtime(ae));
	assertEqualInt(100, archive_entry_mtime_nsec(ae));
	assertEqualInt(0, archive_entry_atime(ae));
	assertEqualInt(0, archive_entry_ctime(ae));
	assertEqualString("file", archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG | 0755, archive_entry_mode(ae));
	assertEqualInt(datasize, archive_entry_size(ae));
	assertEqualIntA(a, datasize, archive_read_data(a, filedata2, datasize));
	assertEqualMem(filedata, filedata2, datasize);

	/* Verify the end of the archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_7ZIP, archive_format(a));

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	free(buff);
	free(filedata);
	free(filedata2);
}

DEFINE_TEST(test_write_format_7zip_large_bzip2)
{
	/* Test that making a 7-Zip archive file with bzip2 compression. */
	test_large("bzip2");
}

DEFINE_TEST(test_write_format_7zip_large_copy)
{
	/* Test that making a 7-Zip archive file without compression. */
	test_large("copy");
}

DEFINE_TEST(test_write_format_7zip_large_deflate)
{
	/* Test that making a 7-Zip archive file with deflate compression. */
	test_large("deflate");
}

DEFINE_TEST(test_write_format_7zip_large_lzma1)
{
	/* Test that making a 7-Zip archive file with lzma1 compression. */
	test_large("lzma1");
}

DEFINE_TEST(test_write_format_7zip_large_lzma2)
{
	/* Test that making a 7-Zip archive file with lzma2 compression. */
	test_large("lzma2");
}

DEFINE_TEST(test_write_format_7zip_large_ppmd)
{
	/* Test that making a 7-Zip archive file with PPMd compression. */
	test_large("ppmd");
}
