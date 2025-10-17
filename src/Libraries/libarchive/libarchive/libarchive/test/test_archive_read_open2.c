/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

static int
open_cb(struct archive *a, void *client)
{
	(void)a; /* UNUSED */
	(void)client; /* UNUSED */
	return 0;
}
static ssize_t
read_cb(struct archive *a, void *client, const void **buff)
{
	(void)a; /* UNUSED */
	(void)client; /* UNUSED */
	(void)buff; /* UNUSED */
	return (ssize_t)0;
}
static int64_t
skip_cb(struct archive *a, void *client, int64_t request)
{
	(void)a; /* UNUSED */
	(void)client; /* UNUSED */
	(void)request; /* UNUSED */
	return (int64_t)0;
}
static int
close_cb(struct archive *a, void *client)
{
	(void)a; /* UNUSED */
	(void)client; /* UNUSED */
	return 0;
}

static void
test(int formatted, archive_open_callback *o, archive_read_callback *r,
    archive_skip_callback *s, archive_close_callback *c,
    int rv, const char *msg)
{
	struct archive* a = archive_read_new();
	if (formatted)
	    assertEqualInt(ARCHIVE_OK,
		archive_read_support_format_empty(a));
	assertEqualInt(rv,
	    archive_read_open2(a, NULL, o, r, s, c));
	assertEqualString(msg, archive_error_string(a));
	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_open2)
{
	const char *no_reader =
	    "No reader function provided to archive_read_open";
	const char *no_formats = "No formats registered";

	test(1, NULL, NULL, NULL, NULL,
	    ARCHIVE_FATAL, no_reader);
	test(1, open_cb, NULL, NULL, NULL,
	    ARCHIVE_FATAL, no_reader);
	test(1, open_cb, read_cb, NULL, NULL,
	    ARCHIVE_OK, NULL);
	test(1, open_cb, read_cb, skip_cb, NULL,
	    ARCHIVE_OK, NULL);
	test(1, open_cb, read_cb, skip_cb, close_cb,
	    ARCHIVE_OK, NULL);
	test(1, NULL, read_cb, skip_cb, close_cb,
	    ARCHIVE_OK, NULL);
	test(1, open_cb, read_cb, skip_cb, NULL,
	    ARCHIVE_OK, NULL);
	test(1, NULL, read_cb, skip_cb, NULL,
	    ARCHIVE_OK, NULL);
	test(1, NULL, read_cb, NULL, NULL,
	    ARCHIVE_OK, NULL);

	test(0, NULL, NULL, NULL, NULL,
	    ARCHIVE_FATAL, no_reader);
	test(0, open_cb, NULL, NULL, NULL,
	    ARCHIVE_FATAL, no_reader);
	test(0, open_cb, read_cb, NULL, NULL,
	    ARCHIVE_FATAL, no_formats);
	test(0, open_cb, read_cb, skip_cb, NULL,
	    ARCHIVE_FATAL, no_formats);
	test(0, open_cb, read_cb, skip_cb, close_cb,
	    ARCHIVE_FATAL, no_formats);
}
