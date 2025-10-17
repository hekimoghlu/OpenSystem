/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

DEFINE_TEST(test_read_filter_uudecode_raw)
{
        struct archive_entry *ae;
        struct archive *a;

	const char *name = "test_read_filter_uudecode_raw.uu";

        assert((a = archive_read_new()) != NULL);
        assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
        assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_raw(a));
        copy_reference_file(name);
        assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, name, 670));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("LICENSE.txt", archive_entry_pathname(ae));
	assertEqualInt((AE_IFREG | 0755), archive_entry_mode(ae));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_filter_uudecode_base64_raw)
{
        struct archive_entry *ae;
        struct archive *a;

	const char *name = "test_read_filter_uudecode_base64_raw.uu";

        assert((a = archive_read_new()) != NULL);
        assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
        assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_raw(a));
        copy_reference_file(name);
        assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, name, 670));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("LICENSE2.txt", archive_entry_pathname(ae));
	assertEqualInt((AE_IFREG | 0600), archive_entry_mode(ae));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
