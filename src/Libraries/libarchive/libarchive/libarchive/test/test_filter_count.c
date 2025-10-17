/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

static void read_test(const char *name);
static void write_test(void);

static void
read_test(const char *name)
{
	struct archive* a = archive_read_new();
	int r;

	r = archive_read_support_filter_bzip2(a);
	if((ARCHIVE_WARN == r && !canBzip2()) || ARCHIVE_WARN > r) {
		skipping("bzip2 unsupported");
		return;
	}

	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));

	extract_reference_file(name);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, name, 2));
	/* bzip2 and none */
	assertEqualInt(2, archive_filter_count(a));
	
	archive_read_free(a);
}

static void
write_test(void)
{
	char buff[4096];
	struct archive* a = archive_write_new();
	int r;

	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_bytes_per_block(a, 10));

	r = archive_write_add_filter_bzip2(a);
	if((ARCHIVE_WARN == r && !canBzip2()) || ARCHIVE_WARN > r) {
		skipping("bzip2 unsupported");
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_write_open_memory(a, buff, 4096, 0));
	/* bzip2 and none */
	assertEqualInt(2, archive_filter_count(a));
	archive_write_free(a);
}

DEFINE_TEST(test_filter_count)
{
	read_test("test_compat_bzip2_1.tbz");
	write_test();
}


