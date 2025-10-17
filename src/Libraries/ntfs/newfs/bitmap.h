/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#ifndef _NTFS_BITMAP_H
#define _NTFS_BITMAP_H

#include "types.h"
#include "attrib.h"

/*
 * NOTES:
 *
 * - Operations are 8-bit only to ensure the functions work both on little
 *   and big endian machines! So don't make them 32-bit ops!
 * - bitmap starts at bit = 0 and ends at bit = bitmap size - 1.
 * - _Caller_ has to make sure that the bit to operate on is less than the
 *   size of the bitmap.
 */

extern void ntfs_bit_set(u8 *bitmap, const u64 bit, const u8 new_value);
extern int  ntfs_bitmap_set_run(ntfs_attr *na, s64 start_bit, s64 count);
extern int  ntfs_bitmap_clear_run(ntfs_attr *na, s64 start_bit, s64 count);

/**
 * ntfs_bitmap_set_bit - set a bit in a bitmap
 * @na:		attribute containing the bitmap
 * @bit:	bit to set
 *
 * Set the @bit in the bitmap described by the attribute @na.
 *
 * On success return 0 and on error return -1 with errno set to the error code.
 */
static __inline__ int ntfs_bitmap_set_bit(ntfs_attr *na, s64 bit)
{
	return ntfs_bitmap_set_run(na, bit, 1);
}

/**
 * ntfs_bitmap_clear_bit - clear a bit in a bitmap
 * @na:		attribute containing the bitmap
 * @bit:	bit to clear
 *
 * Clear @bit in the bitmap described by the attribute @na.
 *
 * On success return 0 and on error return -1 with errno set to the error code.
 */
static __inline__ int ntfs_bitmap_clear_bit(ntfs_attr *na, s64 bit)
{
	return ntfs_bitmap_clear_run(na, bit, 1);
}

#endif /* defined _NTFS_BITMAP_H */

