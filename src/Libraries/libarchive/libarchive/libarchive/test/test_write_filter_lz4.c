/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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
 * A basic exercise of lz4 reading and writing.
 */

DEFINE_TEST(test_write_filter_lz4)
{
	struct archive_entry *ae;
	struct archive* a;
	char *buff, *data;
	size_t buffsize, datasize;
	char path[16];
	size_t used1, used2;
	int i, r, use_prog = 0, filecount;

	assert((a = archive_write_new()) != NULL);
	r = archive_write_add_filter_lz4(a);
	if (archive_liblz4_version() == NULL) {
		if (!canLz4()) {
			skipping("lz4 writing not supported on this platform");
			assertEqualInt(ARCHIVE_WARN, r);
			assertEqualInt(ARCHIVE_OK, archive_write_free(a));
			return;
		} else {
			assertEqualInt(ARCHIVE_WARN, r);
			use_prog = 1;
		}
	} else {
		assertEqualInt(ARCHIVE_OK, r);
	}
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	buffsize = 2000000;
	assert(NULL != (buff = (char *)malloc(buffsize)));

	datasize = 10000;
	assert(NULL != (data = (char *)calloc(datasize, 1)));
	filecount = 10;

	/*
	 * Write a filecount files and read them all back.
	 */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, (use_prog)?ARCHIVE_WARN:ARCHIVE_OK,
	    archive_write_add_filter_lz4(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_bytes_per_block(a, 1024));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_bytes_in_last_block(a, 1024));
	assertEqualInt(ARCHIVE_FILTER_LZ4, archive_filter_code(a, 0));
	assertEqualString("lz4", archive_filter_name(a, 0));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, buffsize, &used1));
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_filetype(ae, AE_IFREG);
	archive_entry_set_size(ae, datasize);
	for (i = 0; i < filecount; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		archive_entry_copy_pathname(ae, path);
		assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
		assertA(datasize
		    == (size_t)archive_write_data(a, data, datasize));
	}
	archive_entry_free(ae);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	r = archive_read_support_filter_lz4(a);
	if (r == ARCHIVE_WARN) {
		skipping("Can't verify lz4 writing by reading back;"
		    " lz4 reading not fully supported on this platform");
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
		goto cleanup;
	}

	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_memory(a, buff, used1));
	for (i = 0; i < filecount; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		if (!assertEqualInt(ARCHIVE_OK,
			archive_read_next_header(a, &ae)))
			break;
		assertEqualString(path, archive_entry_pathname(ae));
		assertEqualInt((int)datasize, archive_entry_size(ae));
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * Repeat the cycle again, this time setting some compression
	 * options.
	 */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_bytes_per_block(a, 10));
	assertEqualIntA(a, (use_prog)?ARCHIVE_WARN:ARCHIVE_OK,
	    archive_write_add_filter_lz4(a));
	assertEqualIntA(a, ARCHIVE_FAILED,
	    archive_write_set_options(a, "lz4:nonexistent-option=0"));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_options(a, "lz4:compression-level=1"));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_filter_option(a, NULL, "compression-level", "9"));
	assertEqualIntA(a, ARCHIVE_FAILED,
	    archive_write_set_filter_option(a, NULL, "compression-level", "abc"));
	assertEqualIntA(a, ARCHIVE_FAILED,
	    archive_write_set_filter_option(a, NULL, "compression-level", "99"));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_options(a, "lz4:compression-level=9"));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, buffsize, &used2));
	for (i = 0; i < filecount; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		assert((ae = archive_entry_new()) != NULL);
		archive_entry_copy_pathname(ae, path);
		archive_entry_set_size(ae, datasize);
		archive_entry_set_filetype(ae, AE_IFREG);
		assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
		assertA(datasize == (size_t)archive_write_data(
		    a, data, datasize));
		archive_entry_free(ae);
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	failure("compression-level=9 wrote %d bytes, default wrote %d bytes",
	    (int)used2, (int)used1);
	assert(used2 < used1);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	r = archive_read_support_filter_lz4(a);
	if (r != ARCHIVE_OK && !use_prog) {
		skipping("lz4 reading not fully supported on this platform");
	} else {
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_support_filter_all(a));
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_open_memory(a, buff, used2));
		for (i = 0; i < filecount; i++) {
			snprintf(path, sizeof(path), "file%03d", i);
			if (!assertEqualInt(ARCHIVE_OK,
				archive_read_next_header(a, &ae)))
				break;
			assertEqualString(path, archive_entry_pathname(ae));
			assertEqualInt((int)datasize, archive_entry_size(ae));
		}
		assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	}
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * Repeat again, with much lower compression.
	 */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_bytes_per_block(a, 10));
	assertEqualIntA(a, (use_prog)?ARCHIVE_WARN:ARCHIVE_OK,
	    archive_write_add_filter_lz4(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_filter_option(a, NULL, "compression-level", "1"));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, buffsize, &used2));
	for (i = 0; i < filecount; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		assert((ae = archive_entry_new()) != NULL);
		archive_entry_copy_pathname(ae, path);
		archive_entry_set_size(ae, datasize);
		archive_entry_set_filetype(ae, AE_IFREG);
		assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
		failure("Writing file %s", path);
		assertEqualIntA(a, datasize,
		    (size_t)archive_write_data(a, data, datasize));
		archive_entry_free(ae);
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

#if 0
	failure("Compression-level=1 wrote %d bytes; default wrote %d bytes",
	    (int)used2, (int)used1);
	assert(used2 > used1);
#endif

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	r = archive_read_support_filter_lz4(a);
	if (r == ARCHIVE_WARN) {
		skipping("lz4 reading not fully supported on this platform");
	} else {
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_open_memory(a, buff, used2));
		for (i = 0; i < filecount; i++) {
			snprintf(path, sizeof(path), "file%03d", i);
			if (!assertEqualInt(ARCHIVE_OK,
				archive_read_next_header(a, &ae)))
				break;
			assertEqualString(path, archive_entry_pathname(ae));
			assertEqualInt((int)datasize, archive_entry_size(ae));
		}
		assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	}
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * Test various premature shutdown scenarios to make sure we
	 * don't crash or leak memory.
	 */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, (use_prog)?ARCHIVE_WARN:ARCHIVE_OK,
	    archive_write_add_filter_lz4(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, (use_prog)?ARCHIVE_WARN:ARCHIVE_OK,
	    archive_write_add_filter_lz4(a));
	assertEqualInt(ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, (use_prog)?ARCHIVE_WARN:ARCHIVE_OK,
	    archive_write_add_filter_lz4(a));
	assertEqualInt(ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, (use_prog)?ARCHIVE_WARN:ARCHIVE_OK,
	    archive_write_add_filter_lz4(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, buffsize, &used2));
	assertEqualInt(ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/*
	 * Clean up.
	 */
cleanup:
	free(data);
	free(buff);
}

static void
test_options(const char *options)
{
	struct archive_entry *ae;
	struct archive* a;
	char *buff, *data;
	size_t buffsize, datasize;
	char path[16];
	size_t used1;
	int i, r, use_prog = 0, filecount;

	assert((a = archive_write_new()) != NULL);
	r = archive_write_add_filter_lz4(a);
	if (archive_liblz4_version() == NULL) {
		if (!canLz4()) {
			skipping("lz4 writing not supported on this platform");
			assertEqualInt(ARCHIVE_WARN, r);
			assertEqualInt(ARCHIVE_OK, archive_write_free(a));
			return;
		} else {
			assertEqualInt(ARCHIVE_WARN, r);
			use_prog = 1;
		}
	} else {
		assertEqualInt(ARCHIVE_OK, r);
	}
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	buffsize = 2000000;
	assert(NULL != (buff = (char *)malloc(buffsize)));

	datasize = 10000;
	assert(NULL != (data = (char *)calloc(datasize, 1)));
	filecount = 10;

	/*
	 * Write a filecount files and read them all back.
	 */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, (use_prog)?ARCHIVE_WARN:ARCHIVE_OK,
	    archive_write_add_filter_lz4(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_options(a, options));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_bytes_per_block(a, 1024));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_bytes_in_last_block(a, 1024));
	assertEqualInt(ARCHIVE_FILTER_LZ4, archive_filter_code(a, 0));
	assertEqualString("lz4", archive_filter_name(a, 0));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, buffsize, &used1));
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_filetype(ae, AE_IFREG);
	archive_entry_set_size(ae, datasize);
	for (i = 0; i < filecount; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		archive_entry_copy_pathname(ae, path);
		assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
		assertA(datasize
		    == (size_t)archive_write_data(a, data, datasize));
	}
	archive_entry_free(ae);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	r = archive_read_support_filter_lz4(a);
	if (r == ARCHIVE_WARN) {
		skipping("Can't verify lz4 writing by reading back;"
		    " lz4 reading not fully supported on this platform");
	} else {
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_open_memory(a, buff, used1));
		for (i = 0; i < filecount; i++) {
			snprintf(path, sizeof(path), "file%03d", i);
			if (!assertEqualInt(ARCHIVE_OK,
				archive_read_next_header(a, &ae)))
				break;
			assertEqualString(path, archive_entry_pathname(ae));
			assertEqualInt((int)datasize, archive_entry_size(ae));
		}
		assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	}
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * Clean up.
	 */
	free(data);
	free(buff);
}

DEFINE_TEST(test_write_filter_lz4_disable_stream_checksum)
{
	test_options("lz4:!stream-checksum");
}

DEFINE_TEST(test_write_filter_lz4_enable_block_checksum)
{
	test_options("lz4:block-checksum");
}

DEFINE_TEST(test_write_filter_lz4_block_size_4)
{
	test_options("lz4:block-size=4");
}

DEFINE_TEST(test_write_filter_lz4_block_size_5)
{
	test_options("lz4:block-size=5");
}

DEFINE_TEST(test_write_filter_lz4_block_size_6)
{
	test_options("lz4:block-size=6");
}

DEFINE_TEST(test_write_filter_lz4_block_dependence)
{
	test_options("lz4:block-dependence");
}

/*
 * TODO: Figure out how to correctly handle this.
 *
 * This option simply fails on some versions of the LZ4 libraries.
 */
/*
XXXDEFINE_TEST(test_write_filter_lz4_block_dependence_hc)
{
	test_options("lz4:block-dependence,lz4:compression-level=9");
}
*/
