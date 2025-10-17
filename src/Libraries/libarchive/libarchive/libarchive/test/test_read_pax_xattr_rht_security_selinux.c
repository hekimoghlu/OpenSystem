/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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

DEFINE_TEST(test_read_pax_xattr_rht_security_selinux)
{
	struct archive *a;
	struct archive_entry *ae;
	const char *refname = "test_read_pax_xattr_rht_security_selinux.tar";
	const char *xname; /* For xattr tests. */
	const void *xval; /* For xattr tests. */
	size_t xsize; /* For xattr tests. */
	const char *string;

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));

	extract_reference_file(refname);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	assertEqualInt(ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(1, archive_entry_xattr_count(ae));
	assertEqualInt(1, archive_entry_xattr_reset(ae));

	assertEqualInt(0, archive_entry_xattr_next(ae, &xname, &xval, &xsize));
	assertEqualString(xname, "security.selinux");
	string = "system_u:object_r:admin_home_t:s0";
	assertEqualMem(xval, string, xsize);
	assertEqualInt((int)xsize, strlen(string));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
