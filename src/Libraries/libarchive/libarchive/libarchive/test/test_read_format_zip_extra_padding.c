/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
 * Test archive verifies that we ignore padding in the extra field.
 *
 * APPNOTE.txt does not provide any provision for padding the extra
 * field, so libarchive used to error when there were unconsumed
 * bytes.  Apparently, some Zip writers do routinely put zero padding
 * in the extra field.
 *
 * The extra fields in this test (for both the local file header
 * and the central directory entry) are formatted as follows:
 *
 *   0000 0000 - unrecognized field with type zero, zero bytes
 *   5554 0900 03d258155cdb58155c - UX field with length 9
 *   0000 0400 00000000 - unrecognized field with type zero, four bytes
 *   000000 - three bytes padding
 *
 * The two valid type zero fields should be skipped and ignored, as
 * should the three bytes padding (which is too short to be a valid
 * extra data object).  If there were no errors and we read the UX
 * field correctly, then we've correctly handled all of the padding
 * fields above.
 */


static void verify(struct archive *a) {
	struct archive_entry *ae;

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("a", archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG | 0664, archive_entry_mode(ae));
	assertEqualInt(0x5c1558d2, archive_entry_mtime(ae));
	assertEqualInt(0, archive_entry_ctime(ae));
	assertEqualInt(0x5c1558db, archive_entry_atime(ae));

	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
}

DEFINE_TEST(test_read_format_zip_extra_padding)
{
	const char *refname = "test_read_format_zip_extra_padding.zip";
	struct archive *a;
	char *p;
	size_t s;

	extract_reference_file(refname);

	/* Verify with seeking reader. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, refname, 7));
	verify(a);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/* Verify with streaming reader. */
	p = slurpfile(&s, "%s", refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory(a, p, s, 3));
	verify(a);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	free(p);
}
