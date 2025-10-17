/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
 * Verify reading entries with POSIX.1e and NFSv4 ACLs from archives created
 * with Solaris tar.
 *
 * This should work on all systems, regardless of whether local filesystems
 * support ACLs or not.
 */

static struct archive_test_acl_t acls0[] = {
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_WRITE |
	    ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE,
	  ARCHIVE_ENTRY_ACL_USER, 71, "lp" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 666, "666" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_WRITE | ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 1000, "1000" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_GROUP_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_MASK, -1, ""},
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_OTHER, -1, "" },
};

static struct archive_test_acl_t acls1[] = {
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_WRITE | ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_WRITE | ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 2, "bin" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_GROUP_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_GROUP, 3, "sys" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_MASK, -1, ""},
	{ ARCHIVE_ENTRY_ACL_TYPE_ACCESS, 0,
	  ARCHIVE_ENTRY_ACL_OTHER, -1, "" },
};

static struct archive_test_acl_t acls2[] = {
	{ ARCHIVE_ENTRY_ACL_TYPE_DEFAULT, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_WRITE | ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, -1 ,"" },
	{ ARCHIVE_ENTRY_ACL_TYPE_DEFAULT, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_WRITE | ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_USER, 2, "bin" },
	{ ARCHIVE_ENTRY_ACL_TYPE_DEFAULT, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_GROUP_OBJ, -1, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_DEFAULT, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_GROUP, 3, "sys" },
	{ ARCHIVE_ENTRY_ACL_TYPE_DEFAULT, ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_WRITE | ARCHIVE_ENTRY_ACL_READ,
	  ARCHIVE_ENTRY_ACL_MASK, -1, ""},
	{ ARCHIVE_ENTRY_ACL_TYPE_DEFAULT, 0,
	  ARCHIVE_ENTRY_ACL_OTHER, -1, "" },
};

static struct archive_test_acl_t acls3[] = {
	{ ARCHIVE_ENTRY_ACL_TYPE_DENY,
	    ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_WRITE_DATA |
	    ARCHIVE_ENTRY_ACL_APPEND_DATA |
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_WRITE_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_WRITE_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_WRITE_ACL |
	    ARCHIVE_ENTRY_ACL_WRITE_OWNER |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_GROUP, 12, "daemon" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_WRITE_DATA |
	    ARCHIVE_ENTRY_ACL_APPEND_DATA |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_GROUP, 2, "bin" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_USER, 4, "adm" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_WRITE_DATA |
	    ARCHIVE_ENTRY_ACL_APPEND_DATA |
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_WRITE_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_WRITE_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_WRITE_ACL |
	    ARCHIVE_ENTRY_ACL_WRITE_OWNER |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, 0, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_GROUP_OBJ, 0, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_EVERYONE, 0, "" },
};

static struct archive_test_acl_t acls4[] = {
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_WRITE_DATA |
	    ARCHIVE_ENTRY_ACL_APPEND_DATA |
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_WRITE_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_WRITE_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_WRITE_ACL |
	    ARCHIVE_ENTRY_ACL_WRITE_OWNER |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE |
	    ARCHIVE_ENTRY_ACL_ENTRY_FILE_INHERIT |
	    ARCHIVE_ENTRY_ACL_ENTRY_DIRECTORY_INHERIT |
	    ARCHIVE_ENTRY_ACL_ENTRY_INHERIT_ONLY,
	  ARCHIVE_ENTRY_ACL_USER, 1100, "1100" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE |
	    ARCHIVE_ENTRY_ACL_ENTRY_FILE_INHERIT |
	    ARCHIVE_ENTRY_ACL_ENTRY_DIRECTORY_INHERIT,
	  ARCHIVE_ENTRY_ACL_GROUP, 4, "adm" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_WRITE_DATA |
	    ARCHIVE_ENTRY_ACL_APPEND_DATA |
	    ARCHIVE_ENTRY_ACL_DELETE_CHILD |
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_WRITE_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_WRITE_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_WRITE_ACL |
	    ARCHIVE_ENTRY_ACL_WRITE_OWNER |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_USER_OBJ, 0, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_EXECUTE |
	    ARCHIVE_ENTRY_ACL_READ_DATA |
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_GROUP_OBJ, 0, "" },
	{ ARCHIVE_ENTRY_ACL_TYPE_ALLOW,
	    ARCHIVE_ENTRY_ACL_READ_ATTRIBUTES |
	    ARCHIVE_ENTRY_ACL_READ_NAMED_ATTRS |
	    ARCHIVE_ENTRY_ACL_READ_ACL |
	    ARCHIVE_ENTRY_ACL_SYNCHRONIZE,
	  ARCHIVE_ENTRY_ACL_EVERYONE, 0, "" },
};

DEFINE_TEST(test_compat_solaris_tar_acl)
{
	char name[] = "test_compat_solaris_tar_acl.tar";
	struct archive *a;
	struct archive_entry *ae;

	/* Read archive file */
	assert(NULL != (a = archive_read_new()));
        assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
        assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
        extract_reference_file(name);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, name,
	    10240));

	/* First item has access ACLs */
	assertA(0 == archive_read_next_header(a, &ae));
	failure("One extended ACL should flag all ACLs to be returned.");
	assertEqualInt(7, archive_entry_acl_reset(ae,
	    ARCHIVE_ENTRY_ACL_TYPE_ACCESS));
	assertEntryCompareAcls(ae, acls0, sizeof(acls0)/sizeof(acls0[0]),
	    ARCHIVE_ENTRY_ACL_TYPE_ACCESS, 0644);
	failure("Basic ACLs should set mode to 0644, not %04o",
	    archive_entry_mode(ae)&0777);
	assert((archive_entry_mode(ae) & 0777) == 0644);

	/* Second item has default and access ACLs */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualInt(6, archive_entry_acl_reset(ae,
	    ARCHIVE_ENTRY_ACL_TYPE_ACCESS));
	assertEntryCompareAcls(ae, acls1, sizeof(acls1)/sizeof(acls1[0]),
	    ARCHIVE_ENTRY_ACL_TYPE_ACCESS, 0750);
	failure("Basic ACLs should set mode to 0750, not %04o",
	    archive_entry_mode(ae)&0777);
	assert((archive_entry_mode(ae) & 0777) == 0750);
	assertEqualInt(6, archive_entry_acl_reset(ae,
	    ARCHIVE_ENTRY_ACL_TYPE_DEFAULT));
	assertEntryCompareAcls(ae, acls2, sizeof(acls2)/sizeof(acls2[0]),
	    ARCHIVE_ENTRY_ACL_TYPE_DEFAULT, 0750);

	/* Third item has NFS4 ACLs */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualInt(6, archive_entry_acl_reset(ae,
	    ARCHIVE_ENTRY_ACL_TYPE_NFS4));
	assertEntryCompareAcls(ae, acls3, sizeof(acls3)/sizeof(acls3[0]),
	    ARCHIVE_ENTRY_ACL_TYPE_NFS4, 0);

	/* Fourth item has NFS4 ACLs and inheritance flags */
	assertA(0 == archive_read_next_header(a, &ae));
	assertEqualInt(5, archive_entry_acl_reset(ae,
	    ARCHIVE_ENTRY_ACL_TYPE_NFS4));
	assertEntryCompareAcls(ae, acls4, sizeof(acls4)/sizeof(acls0[4]),
	    ARCHIVE_ENTRY_ACL_TYPE_NFS4, 0);

	/* Close the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
