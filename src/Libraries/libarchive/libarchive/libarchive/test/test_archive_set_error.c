/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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

static void
test(struct archive *a, int code, const char *msg)
{
	archive_set_error(a, code, "%s", msg);

	assertEqualInt(code, archive_errno(a));
	assertEqualString(msg, archive_error_string(a));
}

DEFINE_TEST(test_archive_set_error)
{
	struct archive* a = archive_read_new();

	/* unlike printf("%s", NULL),
	 * archive_set_error(a, code, "%s", NULL)
	 * segfaults, so it's not tested here */
	test(a, 12, "abcdefgh");
	test(a, 0, "123456");
	test(a, -1, "tuvw");
	test(a, 34, "XYZ");

	archive_read_free(a);
}
