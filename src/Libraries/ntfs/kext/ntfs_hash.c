/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#include <sys/cdefs.h>

#include <sys/errno.h>
#include <sys/kernel_types.h>
#include <sys/malloc.h>
#include <sys/mount.h>
#include <sys/queue.h>
#include <sys/systm.h>
#include <sys/ucred.h>
#include <sys/vnode.h>

#include <libkern/OSAtomic.h>
#include <IOKit/IOLib.h>

#include <kern/locks.h>

#include "ntfs.h"
#include "ntfs_debug.h"
#include "ntfs_hash.h"
#include "ntfs_inode.h"
#include "ntfs_layout.h"
#include "ntfs_types.h"
#include "ntfs_inode.h"
#include "ntfs_volume.h"

/* Structures associated with ntfs inode caching. */
static ntfs_inode_list_head *ntfs_inode_hash_table;
static unsigned long ntfs_inode_hash_mask;

/* A sleeping lock to protect concurrent accesses to the ntfs inode hash. */
lck_mtx_t ntfs_inode_hash_lock;

/**
 * ntfs_inode_hash_init - initialize the ntfs inode hash
 *
 * Initialize the ntfs inode hash.
 */
errno_t ntfs_inode_hash_init(void)
{
	/* Create the ntfs inode hash. */
	ntfs_inode_hash_table = hashinit(desiredvnodes, M_TEMP,
			&ntfs_inode_hash_mask);
	if (!ntfs_inode_hash_table) {
		ntfs_error(NULL, "Failed to allocate ntfs inode hash table.");
		return ENOMEM;
	}
	ntfs_debug("ntfs_inode_hash_mask 0x%lx.", ntfs_inode_hash_mask);
	/* Initialize the ntfs inode hash lock. */
	lck_mtx_init(&ntfs_inode_hash_lock, ntfs_lock_grp, ntfs_lock_attr);
	return 0;
}

/**
 * ntfs_inode_hash_deinit - deinitialize the ntfs inode hash
 *
 * Deinitialize the ntfs inode hash.
 */
void ntfs_inode_hash_deinit(void)
{
	/* Deinitialize the ntfs inode hash lock. */
	lck_mtx_destroy(&ntfs_inode_hash_lock, ntfs_lock_grp);
	/*
	 * Free the ntfs inode hash.
	 *
	 * FIXME: There is no hashdeinit() function so we do it ourselves but
	 * this means that if the implementation of hashinit() changes, this
	 * code will need to be adapted.
	 */
    hashdestroy(ntfs_inode_hash_table, M_TEMP, ntfs_inode_hash_mask);
}

/**
 * ntfs_inode_hash - calculate the hash for a given ntfs inode
 * @vol:	ntfs volume to which the inode belongs
 * @mft_no:	inode number/mft record number of the ntfs inode
 *
 * Return the hash for the ntfs inode with mft record number @mft_no on the
 * volume @vol.
 */
static inline unsigned long ntfs_inode_hash(const ntfs_volume *vol,
		const ino64_t mft_no)
{
	return (vol->dev + mft_no) & ntfs_inode_hash_mask;
}

/**
 * ntfs_inode_hash_list - get the hash bucket list for a given ntfs inode
 * @vol:	ntfs volume to which the inode belongs
 * @mft_no:	inode number/mft record number of the ntfs inode
 *
 * Return the hash bucket list for the ntfs inode with mft record number
 * @mft_no on the volume @vol.
 */
static inline ntfs_inode_list_head *ntfs_inode_hash_list(const ntfs_volume *vol,
		const ino64_t mft_no)
{
	return ntfs_inode_hash_table + ntfs_inode_hash(vol, mft_no);
}

/**
 * ntfs_inode_hash_list_find_nolock - find and return a loaded ntfs inode
 *
 * Search the ntfs inode hash bucket @list for the ntfs inode matching @na and
 * if present return it.  If not present return NULL.
 *
 * Locking: Caller must hold the @ntfs_inode_hash_lock.  Note the lock may be
 *	    dropped if an inode is found to be under reclaim or in the process
 *	    of being loaded, in which cases we drop the lock and wait for the
 *	    inode to be reclaimed/loaded and then we retry the search again.
 */
static inline ntfs_inode *ntfs_inode_hash_list_find_nolock(
		const ntfs_volume *vol, const ntfs_inode_list_head *list,
		const ntfs_attr *na)
{
	ntfs_inode *ni;

	/*
	 * Iterate over all the entries in the hash bucket matching @mp and
	 * @mft_no.  If the ntfs_inode is not in cache, the loop is exited with
	 * @ni set to NULL.
	 */
retry:
	LIST_FOREACH(ni, list, hash) {
		if (ni->vol != vol)
			continue;
		if (!ntfs_inode_test(ni, na))
			continue;
		/*
		 * Make sure that the inode cannot disappear under us at this
		 * point by going to sleep and retrying if it is in the process
		 * of being discarded or allocated.
		 */
		if (NInoReclaim(ni) || NInoAlloc(ni)) {
#ifdef DEBUG
			const char *op;

			if (NInoReclaim(ni))
				op = "reclaim";
			else /* if (NInoAlloc(ni)) */
				op = "allocat";
			ntfs_debug("Inode is being %sed, waiting and "
					"retrying.", op);
#endif
			/* Drops the hash lock. */
			ntfs_inode_wait(ni, &ntfs_inode_hash_lock);
			lck_mtx_lock(&ntfs_inode_hash_lock);
			goto retry;
		}
		/* Found the inode. */
		break;
	}
	return ni;
}

/**
 * ntfs_inode_hash_list_find - find and return a loaded ntfs inode
 *
 * Search the ntfs inode hash bucket @list for the ntfs inode matching @na and
 * if present return it.  If not present return NULL.
 *
 * If the found ntfs inode has a vnode attached, then get an iocount reference
 * on the vnode.
 */
static inline ntfs_inode *ntfs_inode_hash_list_find(const ntfs_volume *vol,
		const ntfs_inode_list_head *list, const ntfs_attr *na)
{
	ntfs_inode *ni;

retry:
	lck_mtx_lock(&ntfs_inode_hash_lock);
	ni = ntfs_inode_hash_list_find_nolock(vol, list, na);
	if (ni) {
		vnode_t vn;
		u32 vn_id = 0;

		// FIXME: If this is an extent inode (i.e. it has no vnode), do
		// we want to take an iocount reference on its base vnode?  If
		// so we would need to make sure to release it when finished
		// with the extent inode. -> Need to do that but only when we
		// start looking up extent inodes from the $MFT pageout code
		// path so that the base inode cannot disappear under us which
		// would also cause the extent ntfs inode to disappear under
		// us.
		vn = ni->vn;
		if (vn)
			vn_id = vnode_vid(vn);
		lck_mtx_unlock(&ntfs_inode_hash_lock);
		if (vn && vnode_getwithvid(vn, vn_id))
			goto retry;
		return ni;
	}
	lck_mtx_unlock(&ntfs_inode_hash_lock);
	return ni;
}

/**
 * ntfs_inode_hash_lookup - find and return a loaded ntfs inode
 *
 * Search the ntfs inode hash for the ntfs inode matching @na and if present
 * return it.  If not present return NULL.
 *
 * If the found ntfs inode has a vnode attached, then get an iocount reference
 * on the vnode.
 */
ntfs_inode *ntfs_inode_hash_lookup(ntfs_volume *vol, const ntfs_attr *na)
{
	ntfs_inode_list_head *list;
	ntfs_inode *ni;

	ntfs_debug("Entering for mft_no 0x%llx, type 0x%x, name_len 0x%x.",
			(unsigned long long)na->mft_no,
			(unsigned)le32_to_cpu(na->type), na->name_len);
	list = ntfs_inode_hash_list(vol, na->mft_no);
	ni = ntfs_inode_hash_list_find(vol, list, na);
	ntfs_debug("Done (ntfs_inode %sfound in cache).", ni ? "" : "not ");
	return ni;
}

/**
 * ntfs_inode_hash_get - find or allocate, and return a loaded ntfs inode
 *
 * Search the ntfs inode hash for the ntfs inode matching @na and if present
 * return it.
 *
 * If the found ntfs inode has a vnode attached, then get an iocount reference
 * on the vnode.
 *
 * If not present, allocate the ntfs inode, add it to the hash, and initialize
 * it before returning it.  The inode will be marked NInoAlloc() and no vnode
 * will be attached yet.
 */
ntfs_inode *ntfs_inode_hash_get(ntfs_volume *vol, const ntfs_attr *na)
{
	ntfs_inode_list_head *list;
	ntfs_inode *ni, *nni;

	ntfs_debug("Entering for mft_no 0x%llx, type 0x%x, name_len 0x%x.",
			(unsigned long long)na->mft_no,
			(unsigned)le32_to_cpu(na->type), na->name_len);
	list = ntfs_inode_hash_list(vol, na->mft_no);
	ni = ntfs_inode_hash_list_find(vol, list, na);
	if (ni) {
		ntfs_debug("Done (ntfs_inode found in cache).");
		return ni;
	}
	/* Not found, allocate a new ntfs_inode and initialize it. */
	nni = IOMallocType(ntfs_inode);
	if (!nni) {
		ntfs_error(vol->mp, "Failed to allocate new ntfs_inode.");
		return nni;
	}
	if (ntfs_inode_init(vol, nni, na)) {
		IOFreeType(nni, ntfs_inode);
		ntfs_error(vol->mp, "Failed to initialize new ntfs_inode.");
		return NULL;
	}
	/*
	 * Take the hash lock and ensure a racing process did not already
	 * allocate the inode by searching for it again in the cache.
	 */
retry:
	lck_mtx_lock(&ntfs_inode_hash_lock);
	ni = ntfs_inode_hash_list_find_nolock(vol, list, na);
	if (ni) {
		/*
		 * Someone else already added the ntfs inode so return that and
		 * throw away ours.
		 */
		vnode_t vn;
		u32 vn_id = 0;

		vn = ni->vn;
		if (vn)
			vn_id = vnode_vid(vn);
		/* Drops the hash lock. */
		ntfs_inode_wait_locked(ni, &ntfs_inode_hash_lock);
		if (vn && vnode_getwithvid(vn, vn_id))
			goto retry;
		IOFreeType(nni, ntfs_inode);
		ntfs_debug("Done (ntfs_inode found in cache - lost race)).");
		return ni;
	}
	/*
	 * We have allocated a new ntfs inode, it is NInoLocked() and
	 * NInoAlloc() and we hold the hash lock so we can now add our inode to
	 * the hash list bucket and drop the hash lock.
	 */
	LIST_INSERT_HEAD(list, nni, hash);
	lck_mtx_unlock(&ntfs_inode_hash_lock);
	/* Add the inode to the list of inodes in the volume. */
	lck_mtx_lock(&vol->inodes_lock);
	LIST_INSERT_HEAD(&vol->inodes, nni, inodes);
	lck_mtx_unlock(&vol->inodes_lock);
	ntfs_debug("Done (new ntfs_inode added to cache).");
	return nni;
}

/**
 * ntfs_inode_hash_rm - remove an ntfs inode from the ntfs inode hash
 * @ni:		ntfs inode to remove from the hash
 *
 * Remove the ntfs inode @ni from the ntfs inode hash.
 */
void ntfs_inode_hash_rm(ntfs_inode *ni)
{
	lck_mtx_lock(&ntfs_inode_hash_lock);
	ntfs_inode_hash_rm_nolock(ni);
	lck_mtx_unlock(&ntfs_inode_hash_lock);
}
