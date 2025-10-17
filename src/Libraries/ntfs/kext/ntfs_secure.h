/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#ifndef _OSX_NTFS_SECURE_H
#define _OSX_NTFS_SECURE_H

#include <sys/errno.h>
#include <sys/ucred.h>
#include <sys/vnode.h>

#include "ntfs_types.h"
#include "ntfs_endian.h"
#include "ntfs_layout.h"
#include "ntfs_volume.h"

__attribute__((visibility("hidden"))) extern SDS_ENTRY *ntfs_file_sds_entry;
__attribute__((visibility("hidden"))) extern SDS_ENTRY *ntfs_dir_sds_entry;
__attribute__((visibility("hidden"))) extern SDS_ENTRY *ntfs_file_sds_entry_old;
__attribute__((visibility("hidden"))) extern SDS_ENTRY *ntfs_dir_sds_entry_old;

/**
 * ntfs_rol32 - rotate a value to the left
 * @x:		value whose bits to rotate to the left
 * @n:		number of bits to rotate @x by
 *
 * Rotate the bits of @x to the left by @n bits.
 *
 * Return the rotated value.
 */
static inline u32 ntfs_rol32(const u32 x, const unsigned n)
{
	return (x << n) | (x >> (32 - n));
}

/**
 * ntfs_security_hash - calculate the hash of a security descriptor
 * @sd:		self-relative security descriptor whose hash to calculate
 * @length:	size in bytes of the security descritor @sd
 *
 * Calculate the hash of the self-relative security descriptor @sd of length
 * @length bytes.
 *
 * This hash is used in the $Secure system file as the primary key for the $SDH
 * index and is also stored in the header of each security descriptor in the
 * $SDS data stream as well as in the index data of both the $SII and $SDH
 * indexes.  In all three cases it forms part of the SDS_ENTRY_HEADER
 * structure.
 *
 * Return the calculated security hash in little endian.
 */
static inline le32 ntfs_security_hash(SECURITY_DESCRIPTOR_RELATIVE *sd,
	const u32 length)
{
	le32 *pos, *end;
	u32 hash;

	pos = (le32*)sd;
	end = (le32*)sd + (length / sizeof(le32));
	for (hash = 0; pos < end; pos++)
		hash = le32_to_cpup(pos) + ntfs_rol32(hash, 3);
	return cpu_to_le32(hash);
}

__private_extern__ errno_t ntfs_default_sds_entries_init(void);

__private_extern__ errno_t ntfs_next_security_id_init(ntfs_volume *vol,
		le32 *next_security_id);

__private_extern__ errno_t ntfs_default_security_id_init(ntfs_volume *vol,
		struct vnode_attr *va);

#endif /* _OSX_NTFS_SECURE_H */
