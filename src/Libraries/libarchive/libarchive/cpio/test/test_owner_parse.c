/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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

#include "../cpio.h"
#include "err.h"

#if !defined(_WIN32)
#define ROOT "root"
static const int root_uids[] = { 0 };
static const int root_gids[] = { 0, 1 };
#elif defined(__CYGWIN__)
/* On cygwin, the Administrator user most likely exists (unless
 * it has been renamed or is in a non-English localization), but
 * its primary group membership depends on how the user set up
 * their /etc/passwd. Likely values are 513 (None), 545 (Users),
 * or 544 (Administrators). Just check for one of those...
 * TODO: Handle non-English localizations... e.g. French 'Administrateur'
 *       Use CreateWellKnownSID() and LookupAccountName()?
 */
#define ROOT "Administrator"
static const int root_uids[] = { 500 };
static const int root_gids[] = { 513, 545, 544 };
#endif

#if defined(ROOT)
static int
int_in_list(int i, const int *l, size_t n)
{
	while (n-- > 0)
		if (*l++ == i)
			return (1);
	failure("%d", i);
	return (0);
}
#endif

DEFINE_TEST(test_owner_parse)
{
#if !defined(ROOT)
	skipping("No uid/gid configuration for this OS");
#else
	int uid, gid;

	assert(NULL == owner_parse(ROOT, &uid, &gid));
	assert(int_in_list(uid, root_uids,
		sizeof(root_uids)/sizeof(root_uids[0])));
	assertEqualInt(-1, gid);


	assert(NULL == owner_parse(ROOT ":", &uid, &gid));
	assert(int_in_list(uid, root_uids,
		sizeof(root_uids)/sizeof(root_uids[0])));
	assert(int_in_list(gid, root_gids,
		sizeof(root_gids)/sizeof(root_gids[0])));

	assert(NULL == owner_parse(ROOT ".", &uid, &gid));
	assert(int_in_list(uid, root_uids,
		sizeof(root_uids)/sizeof(root_uids[0])));
	assert(int_in_list(gid, root_gids,
		sizeof(root_gids)/sizeof(root_gids[0])));

	assert(NULL == owner_parse("111", &uid, &gid));
	assertEqualInt(111, uid);
	assertEqualInt(-1, gid);

	assert(NULL == owner_parse("112:", &uid, &gid));
	assertEqualInt(112, uid);
	/* Can't assert gid, since we don't know gid for user #112. */

	assert(NULL == owner_parse("113.", &uid, &gid));
	assertEqualInt(113, uid);
	/* Can't assert gid, since we don't know gid for user #113. */

	assert(NULL == owner_parse(":114", &uid, &gid));
	assertEqualInt(-1, uid);
	assertEqualInt(114, gid);

	assert(NULL == owner_parse(".115", &uid, &gid));
	assertEqualInt(-1, uid);
	assertEqualInt(115, gid);

	assert(NULL == owner_parse("116:117", &uid, &gid));
	assertEqualInt(116, uid);
	assertEqualInt(117, gid);

	/*
	 * TODO: Lookup current user/group name, build strings and
	 * use those to verify username/groupname lookups for ordinary
	 * users.
	 */

	assert(NULL != owner_parse(":nonexistentgroup", &uid, &gid));
	assert(NULL != owner_parse(ROOT ":nonexistentgroup", &uid, &gid));
	assert(NULL !=
	    owner_parse("nonexistentuser:nonexistentgroup", &uid, &gid));
#endif
}
