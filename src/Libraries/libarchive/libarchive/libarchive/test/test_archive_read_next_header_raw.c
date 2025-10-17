/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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

#define DATA "random garbage for testing purposes"

static const char data[sizeof(DATA)] = DATA;

static void
test(int skip_explicitely)
{
	struct archive* a = archive_read_new();
	struct archive_entry* e;

	assertEqualInt(ARCHIVE_OK, archive_read_support_format_raw(a));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_OK, archive_read_open_memory(a,
	    (void *)(uintptr_t) data, sizeof(data)));
	assertEqualString(NULL, archive_error_string(a));

	assertEqualInt(ARCHIVE_OK, archive_read_next_header(a, &e));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	if (skip_explicitely)
		assertEqualInt(ARCHIVE_OK, archive_read_data_skip(a));

	assertEqualInt(ARCHIVE_EOF, archive_read_next_header(a, &e));
	assertEqualInt(0, archive_errno(a));
	assertEqualString(NULL, archive_error_string(a));

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_next_header_raw)
{
	test(1);
	test(0);
}
