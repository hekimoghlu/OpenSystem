/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
 * A "usual" empty tar archive contains only zero bytes
 * and gets handled by the 'empty' format, not by the 'tar'
 * format.  But there are other kinds of empty tar archives
 * that are true tar archives and handled as such.
 */
DEFINE_TEST(test_read_format_tar_empty_pax)
{
	/*
	 * An archive that only contains a PAX 'g' record
	 * and no real files.  (Git will generate these when
	 * archiving an empty project.)
	 */
	struct archive_entry *ae;
	struct archive *a;
	const char *refname = "test_read_format_tar_empty_pax.tar.Z";

	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualInt(ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualInt(ARCHIVE_FILTER_COMPRESS, archive_filter_code(a, 0));
	assertEqualInt(ARCHIVE_FORMAT_TAR_PAX_INTERCHANGE, archive_format(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
