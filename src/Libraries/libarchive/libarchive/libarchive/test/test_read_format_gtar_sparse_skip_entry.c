/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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

#if defined(__BORLANDC__) || (defined(_MSC_VER) &&  _MSC_VER <= 1300)
# define	LITERAL_LL(x)	x##i64
#else
# define	LITERAL_LL(x)	x##ll
#endif
/*
 * To test skip a sparse file entry, this test does not read file data.
 */
DEFINE_TEST(test_read_format_gtar_sparse_skip_entry)
{
	const char *refname = "test_read_format_gtar_sparse_skip_entry.tar.Z.uu";
	struct archive *a;
	struct archive_entry *ae;
	const void *p;
	size_t s;
	int64_t o;

	copy_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular first file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("a", archive_entry_pathname(ae));
	assertEqualInt(LITERAL_LL(10737418244), archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a),
	    ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Verify regular second file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("b", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a),
	    ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_COMPRESS, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_TAR_PAX_INTERCHANGE,
	    archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));


	/*
	 * Read just one block of a sparse file and skip it.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular first file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("a", archive_entry_pathname(ae));
	assertEqualInt(LITERAL_LL(10737418244), archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a),
	    ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualInt(0, archive_read_data_block(a, &p, &s, &o));
	assertEqualInt(4096, s);
	assertEqualInt(0, o);


	/* Verify regular second file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("b", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a),
	    ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_COMPRESS, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_TAR_PAX_INTERCHANGE,
	    archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

