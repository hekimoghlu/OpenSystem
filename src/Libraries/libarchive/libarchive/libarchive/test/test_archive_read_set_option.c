/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
assertEqualInt(__code, archive_read_set_option(__a, __m, __o, __v))

static void
test(int pristine)
{
	struct archive* a = archive_read_new();
	int known_option_rv = pristine ? ARCHIVE_FAILED : ARCHIVE_OK;

	if (!pristine) {
		archive_read_support_filter_all(a);
		archive_read_support_format_all(a);
        }

	/* NULL and "" denote `no option', so they're ok no matter
	 * what, if any, formats are registered */
	should(a, ARCHIVE_OK, NULL, NULL, NULL);
	should(a, ARCHIVE_OK, "", "", "");

	/* unknown modules and options */
	should(a, ARCHIVE_FAILED, "fubar", "snafu", NULL);
	should(a, ARCHIVE_FAILED, "fubar", "snafu", "betcha");

	/* unknown modules and options */
	should(a, ARCHIVE_FAILED, NULL, "snafu", NULL);
	should(a, ARCHIVE_FAILED, NULL, "snafu", "betcha");

	/* ARCHIVE_OK with iso9660 loaded, ARCHIVE_WARN otherwise */
	should(a, known_option_rv, "iso9660", "joliet", NULL);
	should(a, known_option_rv, "iso9660", "joliet", NULL);
	should(a, known_option_rv, NULL, "joliet", NULL);
	should(a, known_option_rv, NULL, "joliet", NULL);

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_set_option)
{
	test(1);
	test(0);
}
