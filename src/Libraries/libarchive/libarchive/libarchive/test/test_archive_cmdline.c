/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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

#define __LIBARCHIVE_TEST
#include "archive_cmdline_private.h"

DEFINE_TEST(test_archive_cmdline)
{
	struct archive_cmdline *cl;

	/* Command name only. */
	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl, "gzip"));
	assertEqualInt(1, cl->argc);
	assertEqualString("gzip", cl->path);
	assertEqualString("gzip", cl->argv[0]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));

	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl, "gzip "));
	assertEqualInt(1, cl->argc);
	failure("path should not include a space character");
	assertEqualString("gzip", cl->path);
	failure("arg0 should not include a space character");
	assertEqualString("gzip", cl->argv[0]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));

	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl,
	    "/usr/bin/gzip "));
	assertEqualInt(1, cl->argc);
	failure("path should be a full path");
	assertEqualString("/usr/bin/gzip", cl->path);
	failure("arg0 should not be a full path");
	assertEqualString("gzip", cl->argv[0]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));

	/* A command line includes space character. */
	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl, "\"gzip \""));
	assertEqualInt(1, cl->argc);
	failure("path should include a space character");
	assertEqualString("gzip ", cl->path);
	failure("arg0 should include a space character");
	assertEqualString("gzip ", cl->argv[0]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));

	/* A command line includes space character: pattern 2.*/
	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl, "\"gzip \"x"));
	assertEqualInt(1, cl->argc);
	failure("path should include a space character");
	assertEqualString("gzip x", cl->path);
	failure("arg0 should include a space character");
	assertEqualString("gzip x", cl->argv[0]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));

	/* A command line includes space character: pattern 3.*/
	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl,
	    "\"gzip \"x\" s \""));
	assertEqualInt(1, cl->argc);
	failure("path should include a space character");
	assertEqualString("gzip x s ", cl->path);
	failure("arg0 should include a space character");
	assertEqualString("gzip x s ", cl->argv[0]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));

	/* A command line includes space character: pattern 4.*/
	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl,
	    "\"gzip\\\" \""));
	assertEqualInt(1, cl->argc);
	failure("path should include a space character");
	assertEqualString("gzip\" ", cl->path);
	failure("arg0 should include a space character");
	assertEqualString("gzip\" ", cl->argv[0]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));

	/* A command name with a argument. */
	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl, "gzip -d"));
	assertEqualInt(2, cl->argc);
	assertEqualString("gzip", cl->path);
	assertEqualString("gzip", cl->argv[0]);
	assertEqualString("-d", cl->argv[1]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));

	/* A command name with two arguments. */
	assert((cl = __archive_cmdline_allocate()) != NULL);
	if (cl == NULL)
		return;
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_parse(cl, "gzip -d -q"));
	assertEqualInt(3, cl->argc);
	assertEqualString("gzip", cl->path);
	assertEqualString("gzip", cl->argv[0]);
	assertEqualString("-d", cl->argv[1]);
	assertEqualString("-q", cl->argv[2]);
	assertEqualInt(ARCHIVE_OK, __archive_cmdline_free(cl));
}
