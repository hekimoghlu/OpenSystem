/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
 * Verify that the various archive_read_support_* functions
 * return appropriate errors when invoked on the wrong kind of
 * archive handle.
 */

typedef struct archive *constructor(void);
typedef int enabler(struct archive *);
typedef int destructor(struct archive *);

static int format_code = 0;
static int format_code_enabler(struct archive *a)
{
	return archive_read_support_format_by_code(a, format_code);
}

static int format_code_setter(struct archive *a)
{
	return archive_read_set_format(a, format_code);
}

static void
test_success(constructor new_, enabler enable_, destructor free_)
{
	struct archive *a = new_();
	int result = enable_(a);
	if (result == ARCHIVE_WARN) {
		assert(NULL != archive_error_string(a));
		assertEqualIntA(a, -1, archive_errno(a));
	} else {
		assertEqualIntA(a, ARCHIVE_OK, result);
		assert(NULL == archive_error_string(a));
		assertEqualIntA(a, 0, archive_errno(a));
	}
	free_(a);
}

static void
test_failure(constructor new_, enabler enable_, destructor free_)
{
	struct archive *a = new_();
	assertEqualIntA(a, ARCHIVE_FATAL, enable_(a));
	assert(NULL != archive_error_string(a));
	assertEqualIntA(a, -1, archive_errno(a));
	free_(a);
}

static void
test_filter_or_format(enabler enable)
{
	test_success(archive_read_new, enable, archive_read_free);
	test_failure(archive_write_new, enable, archive_write_free);
	test_failure(archive_read_disk_new, enable, archive_read_free);
	test_failure(archive_write_disk_new, enable, archive_write_free);
}

DEFINE_TEST(test_archive_read_support)
{
	test_filter_or_format(archive_read_support_format_7zip);
	test_filter_or_format(archive_read_support_format_all);
	test_filter_or_format(archive_read_support_format_ar);
	test_filter_or_format(archive_read_support_format_cab);
	test_filter_or_format(archive_read_support_format_cpio);
	test_filter_or_format(archive_read_support_format_empty);
	test_filter_or_format(archive_read_support_format_iso9660);
	test_filter_or_format(archive_read_support_format_lha);
	test_filter_or_format(archive_read_support_format_mtree);
	test_filter_or_format(archive_read_support_format_tar);
	test_filter_or_format(archive_read_support_format_xar);
	test_filter_or_format(archive_read_support_format_zip);

	int format_codes[] = {
	    ARCHIVE_FORMAT_CPIO,
	    ARCHIVE_FORMAT_CPIO_POSIX,
	    ARCHIVE_FORMAT_CPIO_BIN_LE,
	    ARCHIVE_FORMAT_CPIO_BIN_BE,
	    ARCHIVE_FORMAT_CPIO_SVR4_NOCRC,
	    ARCHIVE_FORMAT_CPIO_SVR4_CRC,
	    ARCHIVE_FORMAT_CPIO_AFIO_LARGE,
	    ARCHIVE_FORMAT_TAR,
	    ARCHIVE_FORMAT_TAR_USTAR,
	    ARCHIVE_FORMAT_TAR_PAX_INTERCHANGE,
	    ARCHIVE_FORMAT_TAR_PAX_RESTRICTED,
	    ARCHIVE_FORMAT_TAR_GNUTAR,
	    ARCHIVE_FORMAT_ISO9660,
	    ARCHIVE_FORMAT_ISO9660_ROCKRIDGE,
	    ARCHIVE_FORMAT_ZIP,
	    ARCHIVE_FORMAT_EMPTY,
	    ARCHIVE_FORMAT_AR,
	    ARCHIVE_FORMAT_AR_GNU,
	    ARCHIVE_FORMAT_AR_BSD,
	    ARCHIVE_FORMAT_MTREE,
	    ARCHIVE_FORMAT_RAW,
	    ARCHIVE_FORMAT_XAR,
	    ARCHIVE_FORMAT_LHA,
	    ARCHIVE_FORMAT_CAB,
	    ARCHIVE_FORMAT_RAR,
	    ARCHIVE_FORMAT_7ZIP,
	    ARCHIVE_FORMAT_WARC,
	    ARCHIVE_FORMAT_RAR_V5,
	};
	unsigned int i;

	for (i = 0; i < sizeof(format_codes) / sizeof(int); i++) {
		format_code = format_codes[i];
		test_filter_or_format(format_code_enabler);
		test_filter_or_format(format_code_setter);
	}

	test_filter_or_format(archive_read_support_filter_all);
	test_filter_or_format(archive_read_support_filter_bzip2);
	test_filter_or_format(archive_read_support_filter_compress);
	test_filter_or_format(archive_read_support_filter_gzip);
	test_filter_or_format(archive_read_support_filter_lzip);
	test_filter_or_format(archive_read_support_filter_lzma);
	test_filter_or_format(archive_read_support_filter_none);
	test_filter_or_format(archive_read_support_filter_rpm);
	test_filter_or_format(archive_read_support_filter_uu);
	test_filter_or_format(archive_read_support_filter_xz);
}
