/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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
#include <fcntl.h>
#include <unistd.h>
#include <uuid/uuid.h>

#if 1 /* for turdfile code only */
#include <string.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#endif

static int chmodx_syscall(void *obj, uid_t fsowner, gid_t fsgrp, int mode, kauth_filesec_t fsacl);
static int fchmodx_syscall(void *obj, uid_t fsowner, gid_t fsgrp, int mode, kauth_filesec_t fsacl);

static int chmodx1(void *obj,
		   int (* chmod_syscall)(void *obj, uid_t fsowner, gid_t fsgrp, int mode,
				       kauth_filesec_t fsacl),
    		   filesec_t fsec);

/*
 * Chmod interfaces.
 */
int
chmodx_np(const char *path, filesec_t fsec)
{
	return(chmodx1((void *)&path, chmodx_syscall, fsec));
}

int
fchmodx_np(int fd, filesec_t fsec)
{
	return(chmodx1((void *)&fd, fchmodx_syscall, fsec));
}

/*
 * Chmod syscalls.
 */
extern int __chmod_extended(char *, uid_t, gid_t, int, kauth_filesec_t);
extern int __fchmod_extended(int, uid_t, gid_t, int, kauth_filesec_t);

static int
chmodx_syscall(void *obj, uid_t fsowner, gid_t fsgrp, int mode, kauth_filesec_t fsacl)
{
	char *path = *(char **)obj;

	return(__chmod_extended(path, fsowner, fsgrp, mode, fsacl));
}

static int
fchmodx_syscall(void *obj, uid_t fsowner, gid_t fsgrp, int mode, kauth_filesec_t fsacl)
{
	int fd = *(int *)obj;
	return(__fchmod_extended(fd, fsowner, fsgrp, mode, fsacl));
}

/*
 * Chmod internals.
 */
	
static int
chmodx1(void *obj,
    int (chmod_syscall)(void *obj, uid_t fsowner, gid_t fsgrp, int mode, kauth_filesec_t fsacl),
    filesec_t fsec)
{
	uid_t fsowner = KAUTH_UID_NONE;
	gid_t fsgrp = KAUTH_GID_NONE;
	mode_t fsec_mode;
	int fsmode = -1;
	size_t size = 0;
	int fsacl_used = 0;
	int delete_acl = 0;
	kauth_filesec_t fsacl = KAUTH_FILESEC_NONE;
	struct kauth_filesec static_filesec;

	if (fsec == NULL) {
		errno = EINVAL;
		return(-1);
	}
	
	/* regular properties */
	if ((filesec_get_property(fsec, FILESEC_OWNER, &fsowner) != 0) && (errno != ENOENT))
		return(-1);
	if ((filesec_get_property(fsec, FILESEC_GROUP, &fsgrp) != 0) && (errno != ENOENT))
		return(-1);
	if ((filesec_get_property(fsec, FILESEC_MODE, &fsec_mode)) != 0) {
		if (errno != ENOENT)
			return(-1);
	} else {
		/* cast up */
		fsmode = fsec_mode;
	}

	/*
	 * We can set any or all of the ACL and UUIDs, but the two are transported in one
	 * structure.  If we have an ACL, we'll use its allocated structure, otherwise we
	 * need our own.
	 */
	if (((filesec_get_property(fsec, FILESEC_ACL_RAW, &fsacl) != 0) ||
		(filesec_get_property(fsec, FILESEC_ACL_ALLOCSIZE, &size) != 0)) &&
	    (errno != ENOENT))
		return(-1);
	/* caller wants to delete ACL, must remember this */
	if (fsacl == _FILESEC_REMOVE_ACL) {
		delete_acl = 1;
		fsacl = 0;
	}
	
	/* no ACL, use local filesec */
	if (fsacl == KAUTH_FILESEC_NONE) {
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
	if (!fsacl_used) {
		/*
		 * If the caller was trying to remove the ACL, and there are no UUIDs,
		 * we can tell the kernel to completely nuke the whole datastructure.
		 */
		if (delete_acl) {
			fsacl = _FILESEC_REMOVE_ACL;
		} else {
			fsacl = KAUTH_FILESEC_NONE;
		}
	}

	return(chmod_syscall(obj, fsowner, fsgrp, fsmode, fsacl));
}
