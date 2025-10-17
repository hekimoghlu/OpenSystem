/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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

/*
 * Exercise the system-independent portion of the ACL support.
 * Check that archive_entry objects can save and restore POSIX.1e-style
 * ACL data.
 *
 * This should work on all systems, regardless of whether local
 * filesystems support ACLs or not.
 */

static struct archive_test_acl_t acls0[] = {
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, 0, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_GROUP_OBJ, 0, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_WRITE,
	  ARCHIVE_ENTRY_ACL_OTHER, 0, "" },
};

static struct archive_test_acl_t acls1[] = {
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 77, "user77" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_GROUP_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_WRITE,
	  ARCHIVE_ENTRY_ACL_OTHER, -1, "" },
};

static struct archive_test_acl_t acls2[] = {
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE | ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 77, "user77" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, 0,
	  ARCHIVE_ENTRY_ACL_USER, 78, "user78" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_GROUP_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, 0007,
	  ARCHIVE_ENTRY_ACL_GROUP, 78, "group78" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_WRITE | ARCHIVE_ENTRY_ACL_EXECUTE,
	  ARCHIVE_ENTRY_ACL_OTHER, -1, "" },
};

/*
 * NFS4 entry types; attempts to set these on top of POSIX.1e
 * attributes should fail.
 */
static struct archive_test_acl_t acls_nfs4[] = {
	/* NFS4 types */
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 78, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_DENY, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 78, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_AUDIT, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 78, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALARM, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 78, "" },

	/* NFS4 tags */
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_EVERYONE, -1, "" },

	/* NFS4 inheritance markers */
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS,
	  ARCHIVE_ENTRY_ACL_READ | ARCHIVE_ENTRY_ACL_ENTRY_FILE_INHERIT,
	  ARCHIVE_ENTRY_ACL_USER, 79, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS,
	  ARCHIVE_ENTRY_ACL_READ | ARCHIVE_ENTRY_ACL_ENTRY_FILE_INHERIT,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, -1, "" },
};

DEFINE_TEST(test_acl_posix1e)
{
	struct archive_entry *ae;
	int i;

	/* Create a simple archive_entry. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_pathname(ae, "file");
        archive_entry_set_mode(ae, S_IFREG | 0777);

	/* Basic owner/owning group should just update mode bits. */

	/*
	 * Note: This features of libarchive's ACL implementation
	 * shouldn't be relied on and should probably be removed.  It
	 * was done to identify trivial ACLs so we could avoid
	 * triggering unnecessary extensions.  It's better to identify
	 * trivial ACLs at the point they are being read from disk.
	 */
	assertEntrySetAcls(ae, acls0, sizeof(acls0)/sizeof(acls0[0]));
	failure("Basic ACLs shouldn't be stored as extended ACLs");
	assert(0 == archive_entry_acl_reset(ae, ARCHIVE_ENTRY_ACL_TYPE_ACCESS));
	failure("Basic ACLs should set mode to 0142, not %04o",
	    archive_entry_mode(ae)&0777);
	assert((archive_entry_mode(ae) & 0777) == 0142);

	/* With any extended ACL entry, we should read back a full set. */
	assertEntrySetAcls(ae, acls1, sizeof(acls1)/sizeof(acls1[0]));
	failure("One extended ACL should flag all ACLs to be returned.");

	/* Check that entry contains only POSIX.1e types */
	assert((archive_entry_acl_types(ae) &
	    ARCHIVE_ENTRY_ACL_TYPE_NFS4) == 0);
	assert((archive_entry_acl_types(ae) &
	    ARCHIVE_ENTRY_ACL_TYPE_POSIX1E) != 0);

	assert(4 == archive_entry_acl_reset(ae, ARCHIVE_ENTRY_ACL_TYPE_ACCESS));
	assertEntryCompareAcls(ae, acls1, sizeof(acls1)/sizeof(acls1[0]),
	    ARCHIVE_ENTRY_ACL_TYPE_ACCESS, 0142);
	failure("Basic ACLs should set mode to 0142, not %04o",
	    archive_entry_mode(ae)&0777);
	assert((archive_entry_mode(ae) & 0777) == 0142);


	/* A more extensive set of ACLs. */
	assertEntrySetAcls(ae, acls2, sizeof(acls2)/sizeof(acls2[0]));
	assertEqualInt(6, archive_entry_acl_reset(ae, ARCHIVE_ENTRY_ACL_TYPE_ACCESS));
	assertEntryCompareAcls(ae, acls2, sizeof(acls2)/sizeof(acls2[0]),
	    ARCHIVE_ENTRY_ACL_TYPE_ACCESS, 0543);
	failure("Basic ACLs should set mode to 0543, not %04o",
	    archive_entry_mode(ae)&0777);
	assert((archive_entry_mode(ae) & 0777) == 0543);

	/*
	 * Check that clearing ACLs gets rid of them all by repeating
	 * the first test.
	 */
	assertEntrySetAcls(ae, acls0, sizeof(acls0)/sizeof(acls0[0]));
	failure("Basic ACLs shouldn't be stored as extended ACLs");
	assert(0 == archive_entry_acl_reset(ae, ARCHIVE_ENTRY_ACL_TYPE_ACCESS));
	failure("Basic ACLs should set mode to 0142, not %04o",
	    archive_entry_mode(ae)&0777);
	assert((archive_entry_mode(ae) & 0777) == 0142);

	/*
	 * Different types of malformed ACL entries that should
	 * fail when added to existing POSIX.1e ACLs.
	 */
	assertEntrySetAcls(ae, acls2, sizeof(acls2)/sizeof(acls2[0]));
	for (i = 0; i < (int)(sizeof(acls_nfs4)/sizeof(acls_nfs4[0])); ++i) {
		struct archive_test_acl_t *p = &acls_nfs4[i];
		failure("Malformed ACL test #%d", i);
		assertEqualInt(ARCHIVE_FAILED,
		    archive_entry_acl_add_entry(ae,
			p->type, p->permset, p->tag, p->qual, p->name));
		assertEqualInt(6,
		    archive_entry_acl_reset(ae,
			ARCHIVE_ENTRY_ACL_TYPE_ACCESS));
	}
	archive_entry_free(ae);
}
