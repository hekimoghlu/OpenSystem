/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
#ifndef _OSX_NTFS_BITMAP_H
#define _OSX_NTFS_BITMAP_H

#include <sys/errno.h>

#include "ntfs_inode.h"
#include "ntfs_types.h"

__private_extern__ errno_t __ntfs_bitmap_set_bits_in_run(ntfs_inode *ni,
		const s64 start_bit, const s64 count, const u8 value,
		const BOOL is_rollback);

/**
 * ntfs_bitmap_set_bits_in_run - set a run of bits in a bitmap to a value
 * @ni:			ntfs inode describing the bitmap
 * @start_bit:		first bit to set
 * @count:		number of bits to set
 * @value:		value to set the bits to (i.e. 0 or 1)
 *
 * Set @count bits starting at bit @start_bit in the bitmap described by the
 * ntfs inode @ni to @value, where @value is either 0 or 1.
 *
 * Return 0 on success and errno on error.
 *
 * Locking: The caller must hold @ni->lock.
 */
static inline errno_t ntfs_bitmap_set_bits_in_run(ntfs_inode *ni,
		const s64 start_bit, const s64 count, const u8 value)
{
	return __ntfs_bitmap_set_bits_in_run(ni, start_bit, count, value,
			FALSE);
}

/**
 * ntfs_bitmap_set_run - set a run of bits in a bitmap
 * @ni:		ntfs inode describing the bitmap
 * @start_bit:	first bit to set
 * @count:	number of bits to set
 *
 * Set @count bits starting at bit @start_bit in the bitmap described by the
 * ntfs inode @ni.
 *
 * Return 0 on success and errno on error.
 *
 * Locking: The caller must hold @ni->lock.
 */
static inline errno_t ntfs_bitmap_set_run(ntfs_inode *ni, const s64 start_bit,
		const s64 count)
{
	return ntfs_bitmap_set_bits_in_run(ni, start_bit, count, 1);
}

/**
 * ntfs_bitmap_clear_run - clear a run of bits in a bitmap
 * @ni:		ntfs inode describing the bitmap
 * @start_bit:	first bit to clear
 * @count:	number of bits to clear
 *
 * Clear @count bits starting at bit @start_bit in the bitmap described by the
 * ntfs inode @ni.
 *
 * Return 0 on success and errno on error.
 *
 * Locking: The caller must hold @ni->lock.
 */
static inline errno_t ntfs_bitmap_clear_run(ntfs_inode *ni,
		const s64 start_bit, const s64 count)
{
	return ntfs_bitmap_set_bits_in_run(ni, start_bit, count, 0);
}

/**
 * ntfs_bitmap_set_bit - set a bit in a bitmap
 * @ni:		ntfs inode describing the bitmap
 * @bit:	bit to set
 *
 * Set bit @bit in the bitmap described by the ntfs inode @ni.
 *
 * Return 0 on success and errno on error.
 *
 * Locking: The caller must hold @ni->lock.
 */
static inline errno_t ntfs_bitmap_set_bit(ntfs_inode *ni, const s64 bit)
{
	return ntfs_bitmap_set_run(ni, bit, 1);
}

/**
 * ntfs_bitmap_clear_bit - clear a bit in a bitmap
 * @ni:		ntfs inode describing the bitmap
 * @bit:	bit to clear
 *
 * Clear bit @bit in the bitmap described by the ntfs inode @ni.
 *
 * Return 0 on success and errno on error.
 *
 * Locking: The caller must hold @ni->lock.
 */
static inline errno_t ntfs_bitmap_clear_bit(ntfs_inode *ni, const s64 bit)
{
	return ntfs_bitmap_clear_run(ni, bit, 1);
}

#endif /* !_OSX_NTFS_BITMAP_H */
