/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

static void
test_empty_file1(void)
{
	struct archive* a = archive_read_new();

	/* Try opening an empty file with the raw handler. */
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_raw(a));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	/* Raw handler doesn't support empty files. */
	assertEqualInt(ARCHIVE_FATAL, archive_read_open_filename(a, "emptyfile", 0));
	assert(NULL != archive_error_string(a));

	archive_read_free(a);
}

static void
test_empty_file2_check(struct archive* a)
{
	struct archive_entry* e;
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_OK, archive_read_open_filename(a, "emptyfile", 0));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_EOF, archive_read_next_header(a, &e));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	archive_read_free(a);
}

static void
test_empty_file2(void)
{
	struct archive* a = archive_read_new();

	/* Try opening an empty file with raw and empty handlers. */
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_raw(a));
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_empty(a));
	test_empty_file2_check(a);

	a = archive_read_new();
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_by_code(a, ARCHIVE_FORMAT_EMPTY));
	test_empty_file2_check(a);

	a = archive_read_new();
	assertEqualInt(ARCHIVE_OK, archive_read_set_format(a, ARCHIVE_FORMAT_EMPTY));
	test_empty_file2_check(a);
}

static void
test_empty_tarfile(void)
{
	struct archive* a = archive_read_new();
	struct archive_entry* e;

	/* Try opening an empty file with raw and empty handlers. */
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_tar(a));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_OK, archive_read_open_filename(a, "empty.tar", 0));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_EOF, archive_read_next_header(a, &e));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	archive_read_free(a);
}

/* 512 zero bytes. */
static char nulls[512];

DEFINE_TEST(test_archive_read_next_header_empty)
{
	FILE *f;

	/* Create an empty file. */
	f = fopen("emptyfile", "wb");
	fclose(f);

	/* Create a file with 512 zero bytes. */
	f = fopen("empty.tar", "wb");
	assertEqualInt(512, fwrite(nulls, 1, 512, f));
	fclose(f);

	test_empty_file1();
	test_empty_file2();
	test_empty_tarfile();

}
