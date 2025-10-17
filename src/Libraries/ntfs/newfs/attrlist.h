/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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
#ifndef _NTFS_ATTRLIST_H
#define _NTFS_ATTRLIST_H

#include "attrib.h"

extern int ntfs_attrlist_need(ntfs_inode *ni);

extern int ntfs_attrlist_entry_add(ntfs_inode *ni, ATTR_RECORD *attr);
extern int ntfs_attrlist_entry_rm(ntfs_attr_search_ctx *ctx);

/**
 * ntfs_attrlist_mark_dirty - set the attribute list dirty
 * @ni:		ntfs inode which base inode contain dirty attribute list
 *
 * Set the attribute list dirty so it is written out later (at the latest at
 * ntfs_inode_close() time).
 *
 * This function cannot fail.
 */
static __inline__ void ntfs_attrlist_mark_dirty(ntfs_inode *ni)
{
	if (ni->nr_extents == -1)
		NInoAttrListSetDirty(ni->base_ni);
	else
		NInoAttrListSetDirty(ni);
}

#endif /* defined _NTFS_ATTRLIST_H */
