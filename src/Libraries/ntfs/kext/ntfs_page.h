/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#ifndef _OSX_NTFS_PAGE_H
#define _OSX_NTFS_PAGE_H

#include <sys/errno.h>
#include <sys/ubc.h>

#include "ntfs_inode.h"
#include "ntfs_types.h"

__private_extern__ int ntfs_pagein(ntfs_inode *ni, s64 attr_ofs, unsigned size,
		upl_t upl, upl_offset_t upl_ofs, int flags);

__private_extern__ errno_t ntfs_page_map_ext(ntfs_inode *ni, s64 ofs,
		upl_t *upl, upl_page_info_array_t *pl, u8 **kaddr,
		const BOOL uptodate, const BOOL rw);

/**
 * ntfs_page_map - map a page of a vnode into memory
 * @ni:		ntfs inode of which to map a page
 * @ofs:	byte offset into @ni of which to map a page
 * @upl:	destination page list for the page
 * @pl:		destination array of pages containing the page itself
 * @kaddr:	destination pointer for the address of the mapped page contents
 * @rw:		if true we intend to modify the page and if false we do not
 *
 * Map the page corresponding to byte offset @ofs into the ntfs inode @ni into
 * memory and return the page list in @upl, the array of pages containing the
 * page in @pl and the address of the mapped page contents in @kaddr.
 *
 * The page is returned uptodate.
 *
 * The caller must set @rw to true if the page is going to be modified and to
 * false otherwise.
 *
 * Note: @ofs must be page aligned.
 *
 * Locking: - Caller must hold an iocount reference on the vnode of @ni.
 *	    - Caller must hold @ni->lock for reading or writing.
 */
static inline errno_t ntfs_page_map(ntfs_inode *ni, s64 ofs, upl_t *upl,
		upl_page_info_array_t *pl, u8 **kaddr, const BOOL rw)
{
	return ntfs_page_map_ext(ni, ofs, upl, pl, kaddr, TRUE, rw);
}

/**
 * ntfs_page_grab - map a page of a vnode into memory
 * @ni:		ntfs inode of which to map a page
 * @ofs:	byte offset into @ni of which to map a page
 * @upl:	destination page list for the page
 * @pl:		destination array of pages containing the page itself
 * @kaddr:	destination pointer for the address of the mapped page contents
 * @rw:		if true we intend to modify the page and if false we do not
 *
 * Map the page corresponding to byte offset @ofs into the ntfs inode @ni into
 * memory and return the page list in @upl, the array of pages containing the
 * page in @pl and the address of the mapped page contents in @kaddr.
 *
 * The page is returned in whatever state it is obtained from the VM, i.e. it
 * may or may not be uptodate.
 *
 * The caller must set @rw to true if the page is going to be modified and to
 * false otherwise.
 *
 * Note: @ofs must be page aligned.
 *
 * Locking: - Caller must hold an iocount reference on the vnode of @ni.
 *	    - Caller must hold @ni->lock for reading or writing.
 */
static inline errno_t ntfs_page_grab(ntfs_inode *ni, s64 ofs, upl_t *upl,
		upl_page_info_array_t *pl, u8 **kaddr, const BOOL rw)
{
	return ntfs_page_map_ext(ni, ofs, upl, pl, kaddr, FALSE, rw);
}

__private_extern__ void ntfs_page_unmap(ntfs_inode *ni, upl_t upl,
		upl_page_info_array_t pl, const BOOL mark_dirty);

__private_extern__ void ntfs_page_dump(ntfs_inode *ni, upl_t upl,
		upl_page_info_array_t pl);

#endif /* !_OSX_NTFS_PAGE_H */
