/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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

DEFINE_TEST(test_archive_read_close_twice_open_filename)
{
	const char *filename = "empty.file";
	struct archive* a = archive_read_new();

	assertMakeFile(filename, 0644, "");
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_empty(a));
	assertEqualInt(ARCHIVE_OK, archive_read_open_filename(a, filename, 0));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
