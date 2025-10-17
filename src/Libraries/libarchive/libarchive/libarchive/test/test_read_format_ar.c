/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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

DEFINE_TEST(test_read_format_ar)
{
	char buff[64];
	const char reffile[] = "test_read_format_ar.ar";
	struct archive_entry *ae;
	struct archive *a;

	extract_reference_file(reffile);
	assert((a = archive_read_new()) != NULL);
	assertA(0 == archive_read_support_filter_all(a));
	assertA(0 == archive_read_support_format_all(a));
	assertA(0 == archive_read_open_filename(a, reffile, 7));

	/* Filename table.  */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualString("//", archive_entry_pathname(ae));
	assertEqualInt(0, archive_entry_mtime(ae));
	assertEqualInt(0, archive_entry_uid(ae));
	assertEqualInt(0, archive_entry_gid(ae));
	assertEqualInt(0, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* First Entry */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualString("yyytttsssaaafff.o", archive_entry_pathname(ae));
	assertEqualInt(1175465652, archive_entry_mtime(ae));
	assertEqualInt(1001, archive_entry_uid(ae));
	assertEqualInt(0, archive_entry_gid(ae));
	assert(8 == archive_entry_size(ae));
	assertA(8 == archive_read_data(a, buff, 10));
	assertEqualMem(buff, "55667788", 8);
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Second Entry */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualString("gghh.o", archive_entry_pathname(ae));
	assertEqualInt(1175465668, archive_entry_mtime(ae));
	assertEqualInt(1001, archive_entry_uid(ae));
	assertEqualInt(0, archive_entry_gid(ae));
	assert(4 == archive_entry_size(ae));
	assertA(4 == archive_read_data(a, buff, 10));
	assertEqualMem(buff, "3333", 4);
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Third Entry */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualString("hhhhjjjjkkkkllll.o", archive_entry_pathname(ae));
	assertEqualInt(1175465713, archive_entry_mtime(ae));
	assertEqualInt(1001, archive_entry_uid(ae));
	assertEqualInt(0, archive_entry_gid(ae));
	assert(9 == archive_entry_size(ae));
	assertA(9 == archive_read_data(a, buff, 9));
	assertEqualMem(buff, "987654321", 9);

	/* Test EOF */
	assertA(1 == archive_read_next_header(a, &ae));
	assertEqualInt(4, archive_file_count(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
