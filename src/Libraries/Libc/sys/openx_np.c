/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#include <sys/types.h>
#include <sys/acl.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

enum {OPENX, MKFIFOX, MKDIRX};

extern int __open_extended(const char *, int, uid_t, gid_t, int, struct kauth_filesec *);
extern int __mkfifo_extended(const char *, uid_t, gid_t, int, struct kauth_filesec *);
extern int __mkdir_extended(const char *, uid_t, gid_t, int, struct kauth_filesec *);

static int
_mkfilex_np(int opcode, const char *path, int flags, filesec_t fsec)
{
	uid_t owner = KAUTH_UID_NONE;
	gid_t group = KAUTH_GID_NONE;
	mode_t mode = 0;
	size_t size = 0;
	int fsacl_used = 0;
	struct kauth_filesec *fsacl = NULL;
	struct kauth_filesec static_filesec;

	/* handle extended security data */
	if (fsec != NULL) {
		/* fetch basic parameters */
		if ((filesec_get_property(fsec, FILESEC_OWNER, &owner) != 0) && (errno != ENOENT))
			return(-1);
		if ((filesec_get_property(fsec, FILESEC_GROUP, &group) != 0) && (errno != ENOENT))
			return(-1);
		if ((filesec_get_property(fsec, FILESEC_MODE, &mode) != 0) && (errno != ENOENT))
			return(-1);

		/* try to fetch the ACL */
		if (((filesec_get_property(fsec, FILESEC_ACL_RAW, &fsacl) != 0) ||
			(filesec_get_property(fsec, FILESEC_ACL_ALLOCSIZE, &size) != 0)) &&
		    (errno != ENOENT))
			return(-1);

		/* only valid for chmod */
		if (fsacl == _FILESEC_REMOVE_ACL) {
			errno = EINVAL;
			return(-1);
		}

		/* no ACL, use local filesec */
		if (fsacl == NULL) {
			bzero(&static_filesec, sizeof(static_filesec));
			fsacl = &static_filesec;
			fsacl->fsec_magic = KAUTH_FILESEC_MAGIC;
			fsacl->fsec_entrycount = KAUTH_FILESEC_NOACL;
		} else {
			fsacl_used = 1;
		}

		/* grab the owner and group UUID if present */
		if (filesec_get_property(fsec, FILESEC_UUID, &fsacl->fsec_owner) != 0) {
			if (errno != ENOENT)
				return(-1);
			bzero(&fsacl->fsec_owner, sizeof(fsacl->fsec_owner));
		} else {
			fsacl_used = 1;
		}
		if (filesec_get_property(fsec, FILESEC_GRPUUID, &fsacl->fsec_group) != 0) {
			if (errno != ENOENT)
				return(-1);
			bzero(&fsacl->fsec_group, sizeof(fsacl->fsec_group));
		} else {
			fsacl_used = 1;
		}

		/* after all this, if we didn't find anything that needs it, don't pass it in */
		if (!fsacl_used)
			fsacl = NULL;
	}

	switch (opcode) {
	case OPENX:
		return(__open_extended(path, flags, owner, group, mode, fsacl));
	case MKFIFOX:
		return(__mkfifo_extended(path, owner, group, mode, fsacl));
	case MKDIRX:
		return(__mkdir_extended(path, owner, group, mode, fsacl));
	}
	/* should never get here */
	errno = EINVAL;
	return(-1);
}

int
openx_np(const char *path, int flags, filesec_t fsec)
{
	/* optimise for the simple case */
	if (!(flags & O_CREAT) || (fsec == NULL))
		return(open(path, flags));
	return(_mkfilex_np(OPENX, path, flags, fsec));
}

int
mkfifox_np(const char *path, filesec_t fsec)
{
	return(_mkfilex_np(MKFIFOX, path, 0, fsec));
}

int
mkdirx_np(const char *path, filesec_t fsec)
{
	return(_mkfilex_np(MKDIRX, path, 0, fsec));
}
