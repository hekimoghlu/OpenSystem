/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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

static struct archive_entry*
create_archive_entry(void) {
	struct archive_entry *ae;

	assert((ae = archive_entry_new()) != NULL);
        archive_entry_set_atime(ae, 2, 20);
        archive_entry_set_ctime(ae, 4, 40);
        archive_entry_set_mtime(ae, 5, 50);
        archive_entry_copy_pathname(ae, "file");
        archive_entry_set_mode(ae, AE_IFREG | 0755);
        archive_entry_set_nlink(ae, 2);
        archive_entry_set_size(ae, 8);
        archive_entry_xattr_add_entry(ae, "user.data1", "ABCDEFG", 7);
        archive_entry_xattr_add_entry(ae, "user.data2", "XYZ", 3);

	return (ae);
}

DEFINE_TEST(test_pax_xattr_header)
{
	static const char *reffiles[] = {
	    "test_pax_xattr_header_all.tar",
	    "test_pax_xattr_header_libarchive.tar",
	    "test_pax_xattr_header_schily.tar",
	    NULL
	};
	struct archive *a;
	struct archive_entry *ae;

	extract_reference_files(reffiles);

	/* First archive, no options */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, 0, archive_write_set_format_pax(a));
	assertEqualIntA(a, 0, archive_write_add_filter_none(a));
	assertEqualInt(0,
	    archive_write_open_filename(a, "test1.tar"));
	ae = create_archive_entry();
        assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
        archive_entry_free(ae);
        assertEqualIntA(a, 8, archive_write_data(a, "12345678", 9));

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_free(a));

	assertEqualFile("test1.tar","test_pax_xattr_header_all.tar");

	/* Second archive, xattrheader=SCHILY */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, 0, archive_write_set_format_pax(a));
	assertEqualIntA(a, 0, archive_write_add_filter_none(a));
	assertEqualIntA(a, 0, archive_write_set_options(a,
	    "xattrheader=SCHILY")); 
	assertEqualInt(0,
	    archive_write_open_filename(a, "test2.tar"));

	ae = create_archive_entry();
        assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
        archive_entry_free(ae);
        assertEqualIntA(a, 8, archive_write_data(a, "12345678", 9));

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_free(a));

	assertEqualFile("test2.tar","test_pax_xattr_header_schily.tar");

	/* Third archive, xattrheader=LIBARCHIVE */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, 0, archive_write_set_format_pax(a));
	assertEqualIntA(a, 0, archive_write_add_filter_none(a));
	assertEqualIntA(a, 0, archive_write_set_options(a,
	    "xattrheader=LIBARCHIVE")); 
	assertEqualInt(0,
	    archive_write_open_filename(a, "test3.tar"));

	ae = create_archive_entry();
        assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
        archive_entry_free(ae);
        assertEqualIntA(a, 8, archive_write_data(a, "12345678", 9));

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_free(a));

	assertEqualFile("test3.tar","test_pax_xattr_header_libarchive.tar");

	/* Fourth archive, xattrheader=ALL */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, 0, archive_write_set_format_pax(a));
	assertEqualIntA(a, 0, archive_write_add_filter_none(a));
	assertEqualIntA(a, 0, archive_write_set_options(a, "xattrheader=ALL")); 
	assertEqualInt(0,
	    archive_write_open_filename(a, "test4.tar"));

	ae = create_archive_entry();
        assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
        archive_entry_free(ae);
        assertEqualIntA(a, 8, archive_write_data(a, "12345678", 9));

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_free(a));

	assertEqualFile("test4.tar","test_pax_xattr_header_all.tar");
}
