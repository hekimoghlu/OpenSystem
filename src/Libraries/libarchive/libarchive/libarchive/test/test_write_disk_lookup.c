/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
group_cleanup(void *d)
{
	int *mp = d;
	assertEqualInt(*mp, 0x13579);
	*mp = 0x2468;
}

static int64_t
group_lookup(void *d, const char *name, int64_t g)
{
	int *mp = d;

	(void)g; /* UNUSED */

	assertEqualInt(*mp, 0x13579);
	if (strcmp(name, "FOOGROUP"))
		return (1);
	return (73);
}

static void
user_cleanup(void *d)
{
	int *mp = d;
	assertEqualInt(*mp, 0x1234);
	*mp = 0x2345;
}

static int64_t
user_lookup(void *d, const char *name, int64_t u)
{
	int *mp = d;

	(void)u; /* UNUSED */

	assertEqualInt(*mp, 0x1234);
	if (strcmp("FOO", name) == 0)
		return (2);
	return (74);
}

DEFINE_TEST(test_write_disk_lookup)
{
	struct archive *a;
	int gmagic = 0x13579, umagic = 0x1234;
	int64_t id;

	assert((a = archive_write_disk_new()) != NULL);

	/* Default uname/gname lookups always return ID. */
	assertEqualInt(0, archive_write_disk_gid(a, "", 0));
	assertEqualInt(12, archive_write_disk_gid(a, "root", 12));
	assertEqualInt(12, archive_write_disk_gid(a, "wheel", 12));
	assertEqualInt(0, archive_write_disk_uid(a, "", 0));
	assertEqualInt(18, archive_write_disk_uid(a, "root", 18));

	/* Register some weird lookup functions. */
	assertEqualInt(ARCHIVE_OK, archive_write_disk_set_group_lookup(a,
		       &gmagic, &group_lookup, &group_cleanup));
	/* Verify that our new function got called. */
	assertEqualInt(73, archive_write_disk_gid(a, "FOOGROUP", 8));
	assertEqualInt(1, archive_write_disk_gid(a, "NOTFOOGROUP", 8));

	/* De-register. */
	assertEqualInt(ARCHIVE_OK,
		       archive_write_disk_set_group_lookup(a, NULL, NULL, NULL));
	/* Ensure our cleanup function got called. */
	assertEqualInt(gmagic, 0x2468);

	/* Same thing with uname lookup.... */
	assertEqualInt(ARCHIVE_OK, archive_write_disk_set_user_lookup(a,
			   &umagic, &user_lookup, &user_cleanup));
	assertEqualInt(2, archive_write_disk_uid(a, "FOO", 0));
	assertEqualInt(74, archive_write_disk_uid(a, "NOTFOO", 1));
	assertEqualInt(ARCHIVE_OK,
	    archive_write_disk_set_user_lookup(a, NULL, NULL, NULL));
	assertEqualInt(umagic, 0x2345);

	/* Try the standard lookup functions. */
	if (archive_write_disk_set_standard_lookup(a) != ARCHIVE_OK) {
		skipping("standard uname/gname lookup");
	} else {
		/* Try a few common names for group #0. */
		id = archive_write_disk_gid(a, "wheel", 8);
		if (id != 0)
			id = archive_write_disk_gid(a, "root", 8);
		failure("Unable to verify lookup of group #0");
#if defined(_WIN32) && !defined(__CYGWIN__)
		/* Not yet implemented on Windows. */
		assertEqualInt(8, id);
#else
		assertEqualInt(0, id);
#endif

		/* Try a few common names for user #0. */
		id = archive_write_disk_uid(a, "root", 8);
		failure("Unable to verify lookup of user #0");
#if defined(_WIN32) && !defined(__CYGWIN__)
		/* Not yet implemented on Windows. */
		assertEqualInt(8, id);
#else
		assertEqualInt(0, id);
#endif
	}

	/* Deregister again and verify the default lookups again. */
	assertEqualInt(ARCHIVE_OK,
	    archive_write_disk_set_group_lookup(a, NULL, NULL, NULL));
	assertEqualInt(ARCHIVE_OK,
	    archive_write_disk_set_user_lookup(a, NULL, NULL, NULL));
	assertEqualInt(0, archive_write_disk_gid(a, "", 0));
	assertEqualInt(0, archive_write_disk_uid(a, "", 0));

	/* Re-register our custom handlers. */
	gmagic = 0x13579;
	umagic = 0x1234;
	assertEqualInt(ARCHIVE_OK, archive_write_disk_set_group_lookup(a,
			   &gmagic, &group_lookup, &group_cleanup));
	assertEqualInt(ARCHIVE_OK, archive_write_disk_set_user_lookup(a,
		      &umagic, &user_lookup, &user_cleanup));

	/* Destroy the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/* Verify our cleanup functions got called. */
	assertEqualInt(gmagic, 0x2468);
	assertEqualInt(umagic, 0x2345);
}
