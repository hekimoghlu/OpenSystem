/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#ifndef _OSX_NTFS_ATTR_LIST_H
#define _OSX_NTFS_ATTR_LIST_H

#include <sys/errno.h>

#include "ntfs_attr.h"
#include "ntfs_endian.h"
#include "ntfs_inode.h"
#include "ntfs_layout.h"
#include "ntfs_types.h"

__private_extern__ errno_t ntfs_attr_list_is_needed(ntfs_inode *ni,
		ATTR_LIST_ENTRY *skip_entry, BOOL *attr_list_is_needed);

__private_extern__ errno_t ntfs_attr_list_delete(ntfs_inode *ni,
		ntfs_attr_search_ctx *ctx);

__private_extern__ errno_t ntfs_attr_list_add(ntfs_inode *ni, MFT_RECORD *m,
		ntfs_attr_search_ctx *ctx);

__private_extern__ errno_t ntfs_attr_list_sync_shrink(ntfs_inode *ni,
		const unsigned start_ofs, ntfs_attr_search_ctx *ctx);

__private_extern__ errno_t ntfs_attr_list_sync_extend(ntfs_inode *base_ni,
		MFT_RECORD *base_m, unsigned al_ofs,
		ntfs_attr_search_ctx *ctx);

/**
 * ntfs_attr_list_sync - update the attribute list content of an ntfs inode
 * @ni:		base ntfs inode whose attribute list attribugte to update
 * @start_ofs:	byte offset into attribute list attribute from which to write
 * @ctx:	initialized attribute search context
 *
 * Write the attribute list attribute value cached in @ni starting at byte
 * offset @start_ofs into it to the attribute list attribute record (if the
 * attribute list attribute is resident) or to disk as specified by the runlist
 * of the attribute list attribute.
 *
 * This function only works when the attribute list content but not its size
 * has changed.
 *
 * @ctx is an initialized, ready to use attribute search context that we use to
 * look up the attribute list attribute in the mapped, base mft record.
 *
 * Return 0 on success and -errno on error.
 */
static inline int ntfs_attr_list_sync(ntfs_inode *ni, const unsigned start_ofs,
		ntfs_attr_search_ctx *ctx)
{
	return ntfs_attr_list_sync_shrink(ni, start_ofs, ctx);
}

__private_extern__ void ntfs_attr_list_entries_delete(ntfs_inode *ni,
		ATTR_LIST_ENTRY *start_entry, ATTR_LIST_ENTRY *end_entry);

/**
 * ntfs_attr_list_entry_delete - delete an attribute list entry
 * @ni:			base ntfs inode whose attribute list to delete from
 * @target_entry:	attribute list entry to be deleted
 *
 * Delete the attribute list attribute entry @target_entry from the attribute
 * list attribute belonging to the base ntfs inode @ni.
 *
 * This function cannot fail.
 */
static inline void ntfs_attr_list_entry_delete(ntfs_inode *ni,
		ATTR_LIST_ENTRY *target_entry)
{
	ntfs_attr_list_entries_delete(ni, target_entry,
			(ATTR_LIST_ENTRY*)((u8*)target_entry +
			le16_to_cpu(target_entry->length)));
}

#endif /* !_OSX_NTFS_ATTR_LIST_H */
