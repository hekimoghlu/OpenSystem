/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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

DEFINE_TEST(test_read_pax_xattr_schily)
{
	struct archive *a;
	struct archive_entry *ae;
	const char *refname = "test_read_pax_xattr_schily.tar";
	const char *xname; /* For xattr tests. */
	const void *xval; /* For xattr tests. */
	size_t xsize; /* For xattr tests. */
	const char *string, *array;

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));

	extract_reference_file(refname);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	assertEqualInt(ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(2, archive_entry_xattr_count(ae));
	assertEqualInt(2, archive_entry_xattr_reset(ae));

	assertEqualInt(0, archive_entry_xattr_next(ae, &xname, &xval, &xsize));
	assertEqualString(xname, "security.selinux");
	string = "system_u:object_r:unlabeled_t:s0";
	assertEqualString(xval, string);
	/* the xattr's value also contains the terminating \0 */
	assertEqualInt((int)xsize, strlen(string) + 1);

	assertEqualInt(0, archive_entry_xattr_next(ae, &xname, &xval, &xsize));
	assertEqualString(xname, "security.ima");
	assertEqualInt((int)xsize, 265);
	/* we only compare the first 12 bytes */
	array = "\x03\x02\x04\xb0\xe9\xd6\x79\x01\x00\x2b\xad\x1e";
	assertEqualMem(xval, array, 12);

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
