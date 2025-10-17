/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#ifndef HFS_EXTENTS_H_
#define HFS_EXTENTS_H_

#include <stdint.h>
#include <stdbool.h>

#include "hfs_format.h"

#if !HFS_EXTENTS_TEST && !HFS_ALLOC_TEST
#include "hfs_cnode.h"
#include "hfs.h"
#include "BTreesInternal.h"
#endif

typedef struct hfs_ext_iter {
	struct vnode		   *vp;			// If NULL, this is an xattr extent
	BTreeIterator			bt_iter;
	uint8_t					ndx;		// Index in group
	bool					last_in_fork;
	uint32_t				file_block;
	uint32_t				group_block_count;
	HFSPlusExtentRecord		group;
} hfs_ext_iter_t;

errno_t hfs_ext_find(vnode_t vp, off_t offset, hfs_ext_iter_t *iter);

errno_t hfs_ext_replace(hfsmount_t *hfsmp, vnode_t vp,
						uint32_t file_block,
						const HFSPlusExtentDescriptor *repl,
						int count,
						HFSPlusExtentRecord catalog_extents);

bool hfs_ext_iter_is_catalog_extents(hfs_ext_iter_t *iter);

static inline void hfs_ext_copy_rec(const HFSPlusExtentRecord src,
									HFSPlusExtentRecord dst)
{
	memcpy(dst, src, sizeof(HFSPlusExtentRecord));
}

static inline uint32_t hfs_ext_end(const HFSPlusExtentDescriptor *ext)
{
	return ext->startBlock + ext->blockCount;
}

#endif // HFS_EXTENTS_H_
