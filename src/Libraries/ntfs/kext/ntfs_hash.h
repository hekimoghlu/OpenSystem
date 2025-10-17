/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
#ifndef _OSX_NTFS_HASH_H
#define _OSX_NTFS_HASH_H

#include <sys/cdefs.h>
#include <sys/errno.h>
#include <sys/mount.h>
#include <sys/queue.h>

#include <kern/locks.h>

#include "ntfs_inode.h"
#include "ntfs_volume.h"

__attribute__((visibility("hidden"))) extern lck_mtx_t ntfs_inode_hash_lock;

__private_extern__ errno_t ntfs_inode_hash_init(void);
__private_extern__ void ntfs_inode_hash_deinit(void);

__private_extern__ ntfs_inode *ntfs_inode_hash_lookup(ntfs_volume *vol,
		const ntfs_attr *na);
__private_extern__ ntfs_inode *ntfs_inode_hash_get(ntfs_volume *vol,
		const ntfs_attr *na);

/**
 * ntfs_inode_hash_rm_nolock - remove an ntfs inode from the ntfs inode hash
 * @ni:		ntfs inode to remove from the hash
 *
 * Remove the ntfs inode @ni from the ntfs inode hash.
 */
static inline void ntfs_inode_hash_rm_nolock(ntfs_inode *ni)
{
	LIST_REMOVE(ni, hash);
}

__private_extern__ void ntfs_inode_hash_rm(ntfs_inode *ni);

#endif /* !_OSX_NTFS_HASH_H */
