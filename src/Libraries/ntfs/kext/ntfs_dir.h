/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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
#ifndef _OSX_NTFS_DIR_H
#define _OSX_NTFS_DIR_H

#include <sys/errno.h>
#include <sys/uio.h>

#include "ntfs.h"
#include "ntfs_inode.h"
#include "ntfs_layout.h"
#include "ntfs_types.h"

/*
 * ntfs_name is used to return the actual found filename to the caller of
 * ntfs_lookup_inode_by_name() in order for the caller
 * (ntfs_vnops.c::ntfs_vnop_lookup()) to be able to deal with the case
 * sensitive name cache effectively.
 */
typedef struct {
	MFT_REF mref;
	FILENAME_TYPE_FLAGS type;
	u8 len;
	ntfschar name[NTFS_MAX_NAME_LEN];
} ntfs_dir_lookup_name;

/* The little endian Unicode string $I30 as a global constant. */
__attribute__((visibility("hidden"))) extern ntfschar I30[5];

__private_extern__ errno_t ntfs_lookup_inode_by_name(ntfs_inode *dir_ni,
		const ntfschar *uname, const signed uname_len,
		MFT_REF *res_mref, ntfs_dir_lookup_name **res_name);

__private_extern__ errno_t ntfs_readdir(ntfs_inode *dir_ni, uio_t uio,
		int *eofflag, int *numdirent);

__private_extern__ errno_t ntfs_dir_is_empty(ntfs_inode *dir_ni);

__private_extern__ errno_t ntfs_dir_entry_delete(ntfs_inode *dir_ni,
		ntfs_inode *ni, const FILENAME_ATTR *fn, const u32 fn_len);

__private_extern__ errno_t ntfs_dir_entry_add(ntfs_inode *dir_ni,
		const FILENAME_ATTR *fn, const u32 fn_len,
		const leMFT_REF mref);

/**
 * struct _ntfs_dirhint - directory hint structure
 *
 * This is used to store state across directory enumerations, i.e. across calls
 * to ntfs_readdir().
 */
struct _ntfs_dirhint {
	TAILQ_ENTRY(_ntfs_dirhint) link;
	unsigned ofs;
	unsigned time;
	unsigned fn_size;
	FILENAME_ATTR *fn;
};
typedef struct _ntfs_dirhint ntfs_dirhint;

/*
 * NTFS_MAX_DIRHINTS cannot be larger than 63 without reducing
 * NTFS_DIR_POS_MASK, because given the 6-bit tag, at most 63 different tags
 * can exist.  When NTFS_MAX_DIRHINTS is larger than 63, the same list may
 * contain dirhints of the same tag, and a staled dirhint may be returned.
 */
#define NTFS_MAX_DIRHINTS 32
#define NTFS_DIRHINT_TTL 45
#define NTFS_DIR_POS_MASK 0x03ffffff
#define NTFS_DIR_TAG_MASK 0xfc000000
#define NTFS_DIR_TAG_SHIFT 26

__private_extern__ void ntfs_dirhints_put(ntfs_inode *ni, BOOL stale_only);

#endif /* !_OSX_NTFS_DIR_H */
