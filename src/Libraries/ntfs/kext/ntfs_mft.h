/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#ifndef _OSX_NTFS_MFT_H
#define _OSX_NTFS_MFT_H

#include <sys/errno.h>

#include "ntfs_inode.h"
#include "ntfs_layout.h"
#include "ntfs_types.h"
#include "ntfs_volume.h"

__private_extern__ errno_t ntfs_mft_record_map_ext(ntfs_inode *ni,
		MFT_RECORD **m, const BOOL mft_is_locked);

/**
 * ntfs_mft_record_map - map and lock an mft record
 * @ni:		ntfs inode whose mft record to map
 * @m:		destination pointer for the mapped mft record
 *
 * The buffer containing the mft record belonging to the ntfs inode @ni is
 * mapped which on OS X means it is held for exclusive via the BL_BUSY flag in
 * the buffer.  The mapped mft record is returned in *@m.
 *
 * Return 0 on success and errno on error.
 *
 * Note: Caller must hold an iocount reference on the vnode of the base inode
 * of @ni.
 */
static inline errno_t ntfs_mft_record_map(ntfs_inode *ni, MFT_RECORD **m)
{
	return ntfs_mft_record_map_ext(ni, m, FALSE); 
}

__private_extern__ void ntfs_mft_record_unmap(ntfs_inode *ni);

__private_extern__ errno_t ntfs_extent_mft_record_map_ext(ntfs_inode *base_ni,
		MFT_REF mref, ntfs_inode **nni, MFT_RECORD **nm,
		const BOOL mft_is_locked);

/**
 * ntfs_extent_mft_record_map - load an extent inode and attach it to its base
 * @base_ni:	base ntfs inode
 * @mref:	mft reference of the extent inode to load
 * @ext_ni:	on successful return, pointer to the ntfs inode structure
 * @ext_mrec:	on successful return, pointer to the mft record structure
 *
 * Load the extent mft record @mref and attach it to its base inode @base_ni.
 * Return the mapped extent mft record if success.
 *
 * On successful return, @ext_ni contains a pointer to the ntfs inode structure
 * of the mapped extent inode and @ext_mrec contains a pointer to the mft
 * record structure of the mapped extent inode.
 *
 * Note: The caller must hold an iocount reference on the vnode of the base
 * inode.
 */
static inline errno_t ntfs_extent_mft_record_map(ntfs_inode *base_ni,
		MFT_REF mref, ntfs_inode **ext_ni, MFT_RECORD **ext_mrec)
{
	return ntfs_extent_mft_record_map_ext(base_ni, mref, ext_ni, ext_mrec,
			FALSE); 
}

static inline void ntfs_extent_mft_record_unmap(ntfs_inode *ni)
{
	ntfs_mft_record_unmap(ni);
}

__private_extern__ errno_t ntfs_mft_record_sync(ntfs_inode *ni);

__private_extern__ errno_t ntfs_mft_mirror_sync(ntfs_volume *vol,
		const s64 rec_no, const MFT_RECORD *m, const BOOL sync);

__private_extern__ errno_t ntfs_mft_record_alloc(ntfs_volume *vol,
		struct vnode_attr *va, struct componentname *cn,
		ntfs_inode *base_ni, ntfs_inode **new_ni, MFT_RECORD **new_m,
		ATTR_RECORD **new_a);

__private_extern__ errno_t ntfs_extent_mft_record_free(ntfs_inode *base_ni,
		ntfs_inode *ni, MFT_RECORD *m);

#endif /* !_OSX_NTFS_MFT_H */
