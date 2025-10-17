/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
#include <sys/snapshot.h>
#include <sys/attr.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

extern int __fs_snapshot(uint32_t, int, const char *, const char *, void *,
    uint32_t);

int
fs_snapshot_create(int dirfd, const char *name, uint32_t flags)
{
	return __fs_snapshot(SNAPSHOT_OP_CREATE, dirfd, name, NULL, NULL, flags);
}

int
fs_snapshot_list(int dirfd, struct attrlist *alist, void *attrbuf, size_t bufsize,
    uint32_t flags)
{
	if (flags != 0) {
		errno = EINVAL;
		return -1;
	}

	return getattrlistbulk(dirfd, alist, attrbuf, bufsize,
	           FSOPT_LIST_SNAPSHOT);
}

int
fs_snapshot_delete(int dirfd, const char *name, uint32_t flags)
{
	return __fs_snapshot(SNAPSHOT_OP_DELETE, dirfd, name, NULL, NULL, flags);
}

int
fs_snapshot_rename(int dirfd, const char *old, const char *new, uint32_t flags)
{
	return __fs_snapshot(SNAPSHOT_OP_RENAME, dirfd, old, new, NULL, flags);
}

int
fs_snapshot_revert(int dirfd, const char *name, uint32_t flags)
{
	return __fs_snapshot(SNAPSHOT_OP_REVERT, dirfd, name, NULL, NULL, flags);
}

int
fs_snapshot_root(int dirfd, const char *name, uint32_t flags)
{
	return __fs_snapshot(SNAPSHOT_OP_ROOT, dirfd, name, NULL, NULL, flags);
}

int
fs_snapshot_mount(int dirfd, const char *dir, const char *snapshot,
    uint32_t flags)
{
	return __fs_snapshot(SNAPSHOT_OP_MOUNT, dirfd, snapshot, dir,
	           NULL, flags);
}
