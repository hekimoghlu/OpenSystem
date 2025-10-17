/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
 * Verify our ability to read sample files compatibly with 'lz4 -d'.
 *
 * In particular:
 *  * lz4 -d will read multiple lz4 streams, concatenating the output
 *  * lz4 -d will stop at the end of a stream if the following data
 *    doesn't start with a lz4 signature.
 */

/*
 * All of the sample files have the same contents; they're just
 * compressed in different ways.
 */
static void
verify(const char *name, const char *n[])
{
	struct archive_entry *ae;
	struct archive *a;
	int i,r;

	assert((a = archive_read_new()) != NULL);
	r = archive_read_support_filter_lz4(a);
	if (r == ARCHIVE_WARN) {
		skipping("lz4 reading not fully supported on this platform");
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, r);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	copy_reference_file(name);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, name, 200));

	/* Read entries, match up names with list above. */
	for (i = 0; n[i] != NULL; ++i) {
		failure("Could not read file %d (%s) from %s", i, n[i], name);
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_next_header(a, &ae));
		if (r != ARCHIVE_OK) {
			archive_read_free(a);
			return;
		}
		assertEqualString(n[i], archive_entry_pathname(ae));
	}

	/* Verify the end-of-archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify that the format detection worked. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_LZ4);
	assertEqualString(archive_filter_name(a, 0), "lz4");
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR_USTAR);

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}


DEFINE_TEST(test_compat_lz4)
{
	const char *n[7] = { "f1", "f2", "f3", "d1/f1", "d1/f2", "d1/f3", NULL };
	const char *n2[7] = { "xfile", "README", "NEWS", NULL };
	/* This sample has been 'split', each piece compressed separately,
	 * then concatenated.  Lz4 will emit the concatenated result. */
	/* Not supported in libarchive 2.6 and earlier */
	verify("test_compat_lz4_1.tar.lz4.uu", n);
	/* This sample has been compressed as a single stream, but then
	 * some unrelated garbage text has been appended to the end. */
	verify("test_compat_lz4_2.tar.lz4.uu", n);
	/* This sample has been compressed as a legacy stream. */
	verify("test_compat_lz4_3.tar.lz4.uu", n);
	/* This sample has been compressed with -B4 option. */
	verify("test_compat_lz4_B4.tar.lz4.uu", n2);
	/* This sample has been compressed with -B5 option. */
	verify("test_compat_lz4_B5.tar.lz4.uu", n2);
	/* This sample has been compressed with -B6 option. */
	verify("test_compat_lz4_B6.tar.lz4.uu", n2);
	/* This sample has been compressed with -B7 option. */
	verify("test_compat_lz4_B7.tar.lz4.uu", n2);
	/* This sample has been compressed with -B4 and -BD options. */
	verify("test_compat_lz4_B4BD.tar.lz4.uu", n2);
	/* This sample has been compressed with -B5 and -BD options. */
	verify("test_compat_lz4_B5BD.tar.lz4.uu", n2);
	/* This sample has been compressed with -B6 and -BD options. */
	verify("test_compat_lz4_B6BD.tar.lz4.uu", n2);
	/* This sample has been compressed with -B7 and -BD options. */
	verify("test_compat_lz4_B7BD.tar.lz4.uu", n2);
	/* This sample has been compressed with -B4 ,-BD and -BX options. */
	verify("test_compat_lz4_B4BDBX.tar.lz4.uu", n2);
}


