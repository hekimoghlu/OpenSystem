/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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

#include <locale.h>

static void
verify(struct archive *a) {
	struct archive_entry *ae;
	const char *p;

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assert((p = archive_entry_pathname_utf8(ae)) != NULL);
	assertEqualUTF8String(p, "File 1.txt");

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assert((p = archive_entry_pathname_utf8(ae)) != NULL);
#if defined(__APPLE__)
	/* Compare NFD string. */
	assertEqualUTF8String(p, "File 2 - o\xCC\x88.txt");
#else
	/* Compare NFC string. */
	assertEqualUTF8String(p, "File 2 - \xC3\xB6.txt");
#endif

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assert((p = archive_entry_pathname_utf8(ae)) != NULL);
#if defined(__APPLE__)
	/* Compare NFD string. */
	assertEqualUTF8String(p, "File 3 - a\xCC\x88.txt");
#else
	/* Compare NFC string. */
	assertEqualUTF8String(p, "File 3 - \xC3\xA4.txt");
#endif

	/* The CRC of the filename fails, so fall back to CDE. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assert((p = archive_entry_pathname_utf8(ae)) != NULL);
	assertEqualUTF8String(p, "File 4 - xx.txt");

	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
}

DEFINE_TEST(test_read_format_zip_utf8_paths)
{
	const char *refname = "test_read_format_zip_7075_utf8_paths.zip";
	struct archive *a;
	char *p;
	size_t s;

	extract_reference_file(refname);

	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}

	/* Verify with seeking reader. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, refname, 10240));
	verify(a);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_free(a));

	/* Verify with streaming reader. */
	p = slurpfile(&s, "%s", refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory(a, p, s, 31));
	verify(a);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_free(a));
	free(p);
}
