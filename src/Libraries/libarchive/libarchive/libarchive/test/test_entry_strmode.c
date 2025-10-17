/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

DEFINE_TEST(test_entry_strmode)
{
	struct archive_entry *entry;

	assert((entry = archive_entry_new()) != NULL);

	archive_entry_set_mode(entry, AE_IFREG | 0642);
	assertEqualString(archive_entry_strmode(entry), "-rw-r---w- ");

	/* Regular file + hardlink still shows as regular file. */
	archive_entry_set_mode(entry, AE_IFREG | 0644);
	archive_entry_set_hardlink(entry, "link");
	assertEqualString(archive_entry_strmode(entry), "-rw-r--r-- ");

	archive_entry_set_mode(entry, 0640);
	archive_entry_set_hardlink(entry, "link");
	assertEqualString(archive_entry_strmode(entry), "hrw-r----- ");
	archive_entry_set_hardlink(entry, NULL);

	archive_entry_set_mode(entry, AE_IFDIR | 0777);
	assertEqualString(archive_entry_strmode(entry), "drwxrwxrwx ");

	archive_entry_set_mode(entry, AE_IFBLK | 03642);
	assertEqualString(archive_entry_strmode(entry), "brw-r-S-wT ");

	archive_entry_set_mode(entry, AE_IFCHR | 05777);
	assertEqualString(archive_entry_strmode(entry), "crwsrwxrwt ");

	archive_entry_set_mode(entry, AE_IFSOCK | 0222);
	assertEqualString(archive_entry_strmode(entry), "s-w--w--w- ");

	archive_entry_set_mode(entry, AE_IFIFO | 0444);
	assertEqualString(archive_entry_strmode(entry), "pr--r--r-- ");

	archive_entry_set_mode(entry, AE_IFLNK | 04000);
	assertEqualString(archive_entry_strmode(entry), "l--S------ ");

	archive_entry_acl_add_entry(entry, ARCHIVE_ENTRY_ACL_TYPE_ACCESS,
	    0007, ARCHIVE_ENTRY_ACL_GROUP, 78, "group78");
	assertEqualString(archive_entry_strmode(entry), "l--S------+");

	/* Release the experimental entry. */
	archive_entry_free(entry);
}
