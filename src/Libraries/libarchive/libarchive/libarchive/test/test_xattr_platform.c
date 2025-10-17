/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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

DEFINE_TEST(test_xattr_platform)
{
#if !ARCHIVE_XATTR_SUPPORT
	skipping("Extended attributes are not supported on this platform");
#else /* ARCHIVE_XATTR_SUPPORT */
	struct archive *a;
	struct archive_entry *ae;
	const char *name;
	const void *value;
	void *rvalue;
	size_t size, insize;
	int e, r;
	const char *attrname = "user.libarchive.test";
	const char *readval = "readval";
	const char *writeval = "writeval";

	assertMakeFile("readtest", 0644, "a");

	if (!setXattr("readtest", attrname, readval, strlen(readval) + 1)) {
		skipping("Extended attributes are not supported on this "
		    "filesystem");
		return;
	}

	/* Read test */
	assert(NULL != (a = archive_read_disk_new()));
	ae = archive_entry_new();
	assert(ae != NULL);
	archive_entry_set_pathname(ae, "readtest");
	assertEqualInt(ARCHIVE_OK,
		archive_read_disk_entry_from_file(a, ae, -1, NULL));
	e = archive_entry_xattr_reset(ae);
	assert(e > 0);

	r = 0;
	while (archive_entry_xattr_next(ae, &name, &value,
	    &size) == ARCHIVE_OK) {
		if (name != NULL && value != NULL && size > 0 &&
		    strcmp(name, attrname) == 0) {
			failure("Attribute value does not match");
			assertEqualString((const char *)value, readval);
			r = 1;
			break;
		}
	}
	failure("Attribute not found: %s", attrname);
	assertEqualInt(r, 1);

	archive_entry_free(ae);
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	assert(NULL != (a = archive_write_disk_new()));
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_TIME |
	    ARCHIVE_EXTRACT_PERM | ARCHIVE_EXTRACT_XATTR);

	/* Write test */
	ae = archive_entry_new();
	assert(ae != NULL);
	archive_entry_set_pathname(ae, "writetest");
	archive_entry_set_filetype(ae, AE_IFREG);
	archive_entry_set_perm(ae, 0654);
	archive_entry_set_mtime(ae, 123456, 7890);
	archive_entry_set_size(ae, 0);
	archive_entry_xattr_add_entry(ae, attrname, writeval,
	    strlen(writeval) + 1);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	rvalue = getXattr("writetest", attrname, &insize);
	if (assertEqualInt(insize, strlen(writeval) + 1) != 0)
		assertEqualMem(rvalue, writeval, insize);
	free(rvalue);
#endif
}
