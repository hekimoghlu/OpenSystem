/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

#define should(__a, __code, __opts) \
assertEqualInt(__code, archive_read_set_options(__a, __opts))

static void
test(int pristine)
{
	struct archive* a = archive_read_new();
	int halfempty_options_rv = pristine ? ARCHIVE_FAILED : ARCHIVE_OK;
	int known_option_rv = pristine ? ARCHIVE_FAILED : ARCHIVE_OK;

	if (!pristine) {
		archive_read_support_filter_all(a);
		archive_read_support_format_all(a);
	}

	/* NULL and "" denote `no option', so they're ok no matter
	 * what, if any, formats are registered */
	should(a, ARCHIVE_OK, NULL);
	should(a, ARCHIVE_OK, "");

	/* unknown modules and options */
	should(a, ARCHIVE_FAILED, "fubar:snafu");
	assertEqualString("Unknown module name: `fubar'",
	    archive_error_string(a));
	should(a, ARCHIVE_FAILED, "fubar:snafu=betcha");
	assertEqualString("Unknown module name: `fubar'",
	    archive_error_string(a));

	/* unknown modules and options */
	should(a, ARCHIVE_FAILED, "snafu");
	assertEqualString("Undefined option: `snafu'",
	    archive_error_string(a));
	should(a, ARCHIVE_FAILED, "snafu=betcha");
	assertEqualString("Undefined option: `snafu'",
	    archive_error_string(a));

	/* ARCHIVE_OK with iso9660 loaded, ARCHIVE_FAILED otherwise */
	should(a, known_option_rv, "iso9660:joliet");
	if (pristine) {
		assertEqualString("Unknown module name: `iso9660'",
		    archive_error_string(a));
	}
	should(a, known_option_rv, "iso9660:joliet");
	if (pristine) {
		assertEqualString("Unknown module name: `iso9660'",
		    archive_error_string(a));
	}
	should(a, known_option_rv, "joliet");
	if (pristine) {
		assertEqualString("Undefined option: `joliet'",
		    archive_error_string(a));
	}
	should(a, known_option_rv, "!joliet");
	if (pristine) {
		assertEqualString("Undefined option: `joliet'",
		    archive_error_string(a));
	}

	should(a, ARCHIVE_OK, ",");
	should(a, ARCHIVE_OK, ",,");

	should(a, halfempty_options_rv, ",joliet");
	if (pristine) {
		assertEqualString("Undefined option: `joliet'",
		    archive_error_string(a));
	}
	should(a, halfempty_options_rv, "joliet,");
	if (pristine) {
		assertEqualString("Undefined option: `joliet'",
		    archive_error_string(a));
	}

	should(a, ARCHIVE_FAILED, "joliet,snafu");
	if (pristine) {
		assertEqualString("Undefined option: `joliet'",
		    archive_error_string(a));
	} else {
		assertEqualString("Undefined option: `snafu'",
		    archive_error_string(a));
	}

	should(a, ARCHIVE_FAILED, "iso9660:snafu");
	if (pristine) {
		assertEqualString("Unknown module name: `iso9660'",
		    archive_error_string(a));
	} else {
		assertEqualString("Undefined option: `iso9660:snafu'",
		    archive_error_string(a));
	}

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_set_options)
{
	test(1);
	test(0);
}
