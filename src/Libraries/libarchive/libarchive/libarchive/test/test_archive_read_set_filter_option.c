/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

#define should(__a, __code, __m, __o, __v) \
assertEqualInt(__code, archive_read_set_filter_option(__a, __m, __o, __v))

static void
test(int pristine)
{
	struct archive* a = archive_read_new();

	if (!pristine)
		archive_read_support_filter_all(a);

	should(a, ARCHIVE_OK, NULL, NULL, NULL);
	should(a, ARCHIVE_OK, "", "", "");

	should(a, ARCHIVE_FAILED, NULL, "fubar", NULL);
	should(a, ARCHIVE_FAILED, NULL, "fubar", "snafu");
	should(a, ARCHIVE_FAILED, "fubar", "snafu", NULL);
	should(a, ARCHIVE_FAILED, "fubar", "snafu", "betcha");

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_set_filter_option)
{
	test(1);
	test(0);
}
