/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
 * The pax size attribute can be used to override the size.
 * It should be validated the same way the normal size is validated.
 * The test data is fuzzer output from
 * https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=48467 .
 */
DEFINE_TEST(test_read_format_tar_invalid_pax_size)
{
	/*
	 * An archive that contains a PAX 'size' record with a large negative value.
	 */
	struct archive_entry *ae;
	struct archive *a;
	const char *refname = "test_read_format_tar_invalid_pax_size.tar";

	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualInt(ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, refname, 10240));
	/* This assert will pass a normal debug build without the pax size check. */
	/* Run this test with `-fsanitize=undefined` to verify.                   */
	assertEqualIntA(a, ARCHIVE_FATAL, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
