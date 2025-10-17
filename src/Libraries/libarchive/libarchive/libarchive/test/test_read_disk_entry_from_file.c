/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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

static const char *
gname_lookup(void *d, int64_t g)
{
	(void)d; /* UNUSED */
	(void)g; /* UNUSED */
	return ("FOOGROUP");
}

static const char *
uname_lookup(void *d, int64_t u)
{
	(void)d; /* UNUSED */
	(void)u; /* UNUSED */
	return ("FOO");
}

DEFINE_TEST(test_read_disk_entry_from_file)
{
	struct archive *a;
	struct archive_entry *entry;
	FILE *f;

	assert((a = archive_read_disk_new()) != NULL);

	assertEqualInt(ARCHIVE_OK, archive_read_disk_set_uname_lookup(a,
			   NULL, &uname_lookup, NULL));
	assertEqualInt(ARCHIVE_OK, archive_read_disk_set_gname_lookup(a,
			   NULL, &gname_lookup, NULL));
	assertEqualString(archive_read_disk_uname(a, 0), "FOO");
	assertEqualString(archive_read_disk_gname(a, 0), "FOOGROUP");

	/* Create a file on disk. */
	f = fopen("foo", "wb");
	assert(f != NULL);
	assertEqualInt(4, fwrite("1234", 1, 4, f));
	fclose(f);

	/* Use archive_read_disk_entry_from_file to get information about it. */
	entry = archive_entry_new();
	assert(entry != NULL);
	archive_entry_copy_pathname(entry, "foo");
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_disk_entry_from_file(a, entry, -1, NULL));

	/* Verify the information we got back. */
	assertEqualString(archive_entry_uname(entry), "FOO");
	assertEqualString(archive_entry_gname(entry), "FOOGROUP");
	assertEqualInt(archive_entry_size(entry), 4);

	/* Destroy the archive. */
	archive_entry_free(entry);
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
