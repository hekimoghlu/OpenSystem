/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#include "types.h"
#include "layout.h"
#include "attrib.h"
#include "attrlist.h"
#include "debug.h"
#include "unistr.h"
#include "logging.h"
#include "misc.h"

/**
 * ntfs_attrlist_need - check whether inode need attribute list
 * @ni:		opened ntfs inode for which perform check
 *
 * Check whether all are attributes belong to one MFT record, in that case
 * attribute list is not needed.
 *
 * Return 1 if inode need attribute list, 0 if not, -1 on error with errno set
 * to the error code. If function succeed errno set to 0. The following error
 * codes are defined:
 *	EINVAL	- Invalid arguments passed to function or attribute haven't got
 *		  attribute list.
 */
int ntfs_attrlist_need(ntfs_inode *ni)
{
	ATTR_LIST_ENTRY *ale;

	if (!ni) {
		ntfs_log_trace("Invalid arguments.\n");
		errno = EINVAL;
		return -1;
	}

	ntfs_log_trace("Entering for inode 0x%llx.\n", (long long) ni->mft_no);

	if (!NInoAttrList(ni)) {
		ntfs_log_trace("Inode haven't got attribute list.\n");
		errno = EINVAL;
		return -1;
	}

	if (!ni->attr_list) {
		ntfs_log_trace("Corrupt in-memory struct.\n");
		errno = EINVAL;
		return -1;
	}

	errno = 0;
	ale = (ATTR_LIST_ENTRY *)ni->attr_list;
	while ((u8*)ale < ni->attr_list + ni->attr_list_size) {
		if (MREF_LE(ale->mft_reference) != ni->mft_no)
			return 1;
		ale = (ATTR_LIST_ENTRY *)((u8*)ale + le16_to_cpu(ale->length));
	}
	return 0;
}

/**
 * ntfs_attrlist_entry_add - add an attribute list attribute entry
 * @ni:		opened ntfs inode, which contains that attribute
 * @attr:	attribute record to add to attribute list
 *
 * Return 0 on success and -1 on error with errno set to the error code. The
 * following error codes are defined:
 *	EINVAL	- Invalid arguments passed to function.
 *	ENOMEM	- Not enough memory to allocate necessary buffers.
 *	EIO	- I/O error occurred or damaged filesystem.
 *	EEXIST	- Such attribute already present in attribute list.
 */
int ntfs_attrlist_entry_add(ntfs_inode *ni, ATTR_RECORD *attr)
{
	ATTR_LIST_ENTRY *ale;
	MFT_REF mref;
	ntfs_attr *na = NULL;
	ntfs_attr_search_ctx *ctx;
	u8 *new_al;
	int entry_len, entry_offset, err;

	ntfs_log_trace("Entering for inode 0x%llx, attr 0x%x.\n",
			(long long) ni->mft_no,
			(unsigned) le32_to_cpu(attr->type));

	if (!ni || !attr) {
		ntfs_log_trace("Invalid arguments.\n");
		errno = EINVAL;
		return -1;
	}

	mref = MK_LE_MREF(ni->mft_no, le16_to_cpu(ni->mrec->sequence_number));

	if (ni->nr_extents == -1)
		ni = ni->base_ni;

	if (!NInoAttrList(ni)) {
		ntfs_log_trace("Attribute list isn't present.\n");
		errno = ENOENT;
		return -1;
	}

	/* Determine size and allocate memory for new attribute list. */
	entry_len = (sizeof(ATTR_LIST_ENTRY) + sizeof(ntfschar) *
			attr->name_length + 7) & ~7;
	new_al = ntfs_calloc(ni->attr_list_size + entry_len);
	if (!new_al)
		return -1;

	/* Find place for the new entry. */
	ctx = ntfs_attr_get_search_ctx(ni, NULL);
	if (!ctx) {
		err = errno;
		goto err_out;
	}
	if (!ntfs_attr_lookup(attr->type, (attr->name_length) ? (ntfschar*)
			((u8*)attr + le16_to_cpu(attr->name_offset)) :
			AT_UNNAMED, attr->name_length, CASE_SENSITIVE,
			(attr->non_resident) ? le64_to_cpu(attr->lowest_vcn) :
			0, (attr->non_resident) ? NULL : ((u8*)attr +
			le16_to_cpu(attr->value_offset)), (attr->non_resident) ?
			0 : le32_to_cpu(attr->value_length), ctx)) {
		/* Found some extent, check it to be before new extent. */
		if (ctx->al_entry->lowest_vcn == attr->lowest_vcn) {
			err = EEXIST;
			ntfs_log_trace("Such attribute already present in the "
					"attribute list.\n");
			ntfs_attr_put_search_ctx(ctx);
			goto err_out;
		}
		/* Add new entry after this extent. */
		ale = (ATTR_LIST_ENTRY*)((u8*)ctx->al_entry +
				le16_to_cpu(ctx->al_entry->length));
	} else {
		/* Check for real errors. */
		if (errno != ENOENT) {
			err = errno;
			ntfs_log_trace("Attribute lookup failed.\n");
			ntfs_attr_put_search_ctx(ctx);
			goto err_out;
		}
		/* No previous extents found. */
		ale = ctx->al_entry;
	}
	/* Don't need it anymore, @ctx->al_entry points to @ni->attr_list. */
	ntfs_attr_put_search_ctx(ctx);

	/* Determine new entry offset. */
	entry_offset = ((u8 *)ale - ni->attr_list);
	/* Set pointer to new entry. */
	ale = (ATTR_LIST_ENTRY *)(new_al + entry_offset);
	/* Zero it to fix valgrind warning. */
	memset(ale, 0, entry_len);
	/* Form new entry. */
	ale->type = attr->type;
	ale->length = cpu_to_le16(entry_len);
	ale->name_length = attr->name_length;
	ale->name_offset = offsetof(ATTR_LIST_ENTRY, name);
	if (attr->non_resident)
		ale->lowest_vcn = attr->lowest_vcn;
	else
		ale->lowest_vcn = 0;
	ale->mft_reference = mref;
	ale->instance = attr->instance;
	memcpy(ale->name, (u8 *)attr + le16_to_cpu(attr->name_offset),
			attr->name_length * sizeof(ntfschar));

	/* Resize $ATTRIBUTE_LIST to new length. */
	na = ntfs_attr_open(ni, AT_ATTRIBUTE_LIST, AT_UNNAMED, 0);
	if (!na) {
		err = errno;
		ntfs_log_trace("Failed to open $ATTRIBUTE_LIST attribute.\n");
		goto err_out;
	}
	if (ntfs_attr_truncate(na, ni->attr_list_size + entry_len)) {
		err = errno;
		ntfs_log_trace("$ATTRIBUTE_LIST resize failed.\n");
		goto err_out;
	}

	/* Copy entries from old attribute list to new. */
	memcpy(new_al, ni->attr_list, entry_offset);
	memcpy(new_al + entry_offset + entry_len, ni->attr_list +
			entry_offset, ni->attr_list_size - entry_offset);

	/* Set new runlist. */
	free(ni->attr_list);
	ni->attr_list = new_al;
	ni->attr_list_size = ni->attr_list_size + entry_len;
	NInoAttrListSetDirty(ni);
	/* Done! */
	ntfs_attr_close(na);
	return 0;
err_out:
	if (na)
		ntfs_attr_close(na);
	free(new_al);
	errno = err;
	return -1;
}

/**
 * ntfs_attrlist_entry_rm - remove an attribute list attribute entry
 * @ctx:	attribute search context describing the attribute list entry
 *
 * Remove the attribute list entry @ctx->al_entry from the attribute list.
 *
 * Return 0 on success and -1 on error with errno set to the error code.
 */
int ntfs_attrlist_entry_rm(ntfs_attr_search_ctx *ctx)
{
	u8 *new_al;
	int new_al_len;
	ntfs_inode *base_ni;
	ntfs_attr *na;
	ATTR_LIST_ENTRY *ale;
	int err;

	if (!ctx || !ctx->ntfs_ino || !ctx->al_entry) {
		ntfs_log_trace("Invalid arguments.\n");
		errno = EINVAL;
		return -1;
	}

	if (ctx->base_ntfs_ino)
		base_ni = ctx->base_ntfs_ino;
	else
		base_ni = ctx->ntfs_ino;
	ale = ctx->al_entry;

	ntfs_log_trace("Entering for inode 0x%llx, attr 0x%x, lowest_vcn %lld.\n",
			(long long) ctx->ntfs_ino->mft_no,
			(unsigned) le32_to_cpu(ctx->al_entry->type),
			(long long) le64_to_cpu(ctx->al_entry->lowest_vcn));

	if (!NInoAttrList(base_ni)) {
		ntfs_log_trace("Attribute list isn't present.\n");
		errno = ENOENT;
		return -1;
	}

	/* Allocate memory for new attribute list. */
	new_al_len = base_ni->attr_list_size - le16_to_cpu(ale->length);
	new_al = ntfs_calloc(new_al_len);
	if (!new_al)
		return -1;

	/* Resize $ATTRIBUTE_LIST to new length. */
	na = ntfs_attr_open(base_ni, AT_ATTRIBUTE_LIST, AT_UNNAMED, 0);
	if (!na) {
		err = errno;
		ntfs_log_trace("Failed to open $ATTRIBUTE_LIST attribute.\n");
		goto err_out;
	}
	if (ntfs_attr_truncate(na, new_al_len)) {
		err = errno;
		ntfs_log_trace("$ATTRIBUTE_LIST resize failed.\n");
		goto err_out;
	}

	/* Copy entries from old attribute list to new. */
	memcpy(new_al, base_ni->attr_list, (u8*)ale - base_ni->attr_list);
	memcpy(new_al + ((u8*)ale - base_ni->attr_list), (u8*)ale + le16_to_cpu(
		ale->length), new_al_len - ((u8*)ale - base_ni->attr_list));

	/* Set new runlist. */
	free(base_ni->attr_list);
	base_ni->attr_list = new_al;
	base_ni->attr_list_size = new_al_len;
	NInoAttrListSetDirty(base_ni);
	/* Done! */
	ntfs_attr_close(na);
	return 0;
err_out:
	if (na)
		ntfs_attr_close(na);
	free(new_al);
	errno = err;
	return -1;
}
