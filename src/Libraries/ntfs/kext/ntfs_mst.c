/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#include <sys/errno.h>

#include "ntfs.h"
#include "ntfs_endian.h"
#include "ntfs_layout.h"
#include "ntfs_mst.h"
#include "ntfs_types.h"

/**
 * ntfs_mst_fixup_post_read - deprotect multi sector transfer protected data
 * @b:		pointer to the data to deprotect
 * @size:	size in bytes of @b
 *
 * Perform the necessary post read multi sector transfer fixup and detect the
 * presence of incomplete multi sector transfers.
 *
 * In the case of an incomplete transfer being detected, overwrite the magic of
 * the ntfs record header being processed with "BAAD" and abort processing.
 *
 * Return 0 on success and EIO on error ("BAAD" magic will be present).
 *
 * NOTE: We consider the absence / invalidity of an update sequence array to
 * mean that the structure is not protected at all and hence does not need to
 * be fixed up.  Thus, we return success and not failure in this case.  This is
 * in contrast to ntfs_mst_fixup_pre_write(), see below.
 */
errno_t ntfs_mst_fixup_post_read(NTFS_RECORD *b, const u32 size)
{
	u16 *usa_pos, *data_pos;
	u16 usa_ofs, usa_count, usn;

	/* Setup the variables. */
	usa_ofs = le16_to_cpu(b->usa_ofs);
	/* Decrement usa_count to get number of fixups. */
	usa_count = le16_to_cpu(b->usa_count) - 1;
	/* Size and alignment checks. */
	if (size & (NTFS_BLOCK_SIZE - 1) || usa_ofs & 1 ||
			(u32)usa_ofs + ((u32)usa_count * 2) > size ||
			(size >> NTFS_BLOCK_SIZE_SHIFT) != usa_count)
		return 0;
	/* Position of usn in update sequence array. */
	usa_pos = (u16*)b + usa_ofs/sizeof(u16);
	/*
	 * The update sequence number which has to be equal to each of the u16
	 * values before they are fixed up.  Note no need to care for
	 * endianness since we are comparing and moving data for on disk
	 * structures which means the data is consistent.  If it is consistenty
	 * the wrong endianness it does not make any difference.
	 */
	usn = *usa_pos;
	/* Position in protected data of first u16 that needs fixing up. */
	data_pos = (u16*)b + NTFS_BLOCK_SIZE/sizeof(u16) - 1;
	/* Check for incomplete multi sector transfer(s). */
	while (usa_count--) {
		if (*data_pos != usn) {
			/*
			 * Incomplete multi sector transfer detected!  )-:
			 * Set the magic to "BAAD" and return failure.
			 * Note that magic_BAAD is already little endian.
			 */
			b->magic = magic_BAAD;
			return EIO;
		}
		data_pos += NTFS_BLOCK_SIZE/sizeof(u16);
	}
	/* Re-setup the variables. */
	usa_count = le16_to_cpu(b->usa_count) - 1;
	data_pos = (u16*)b + NTFS_BLOCK_SIZE/sizeof(u16) - 1;
	/* Fixup all sectors. */
	while (usa_count--) {
		/*
		 * Increment position in usa and restore original data from
		 * the usa into the data buffer.
		 */
		*data_pos = *(++usa_pos);
		/* Increment position in data as well. */
		data_pos += NTFS_BLOCK_SIZE/sizeof(u16);
	}
	return 0;
}

/**
 * ntfs_mst_fixup_pre_write - apply multi sector transfer protection
 * @b:		pointer to the data to protect
 * @size:	size in bytes of @b
 *
 * Perform the necessary pre write multi sector transfer fixup on the data
 * pointed to by @b of @size.
 *
 * Return 0 if fixup applied (success) or EINVAL if no fixup was performed
 * (assumed not needed).  This is in contrast to ntfs_mst_fixup_post_read()
 * above.
 *
 * NOTE: We consider the absence / invalidity of an update sequence array to
 * mean that the structure is not subject to protection and hence does not need
 * to be fixed up.  This means that you have to create a valid update sequence
 * array header in the ntfs record before calling this function, otherwise it
 * will fail (the header needs to contain the position of the update sequence
 * array together with the number of elements in the array).  You also need to
 * initialise the update sequence number before calling this function
 * otherwise a random word will be used (whatever was in the record at that
 * position at that time).
 */
errno_t ntfs_mst_fixup_pre_write(NTFS_RECORD *b, const u32 size)
{
	le16 *usa_pos, *data_pos;
	u16 usa_ofs, usa_count, usn;
	le16 le_usn;

	/* Sanity check + only fixup if it makes sense. */
	if (!b || ntfs_is_baad_record(b->magic) ||
			ntfs_is_hole_record(b->magic))
		return EINVAL;
	/* Setup the variables. */
	usa_ofs = le16_to_cpu(b->usa_ofs);
	/* Decrement usa_count to get number of fixups. */
	usa_count = le16_to_cpu(b->usa_count) - 1;
	/* Size and alignment checks. */
	if (size & (NTFS_BLOCK_SIZE - 1) || usa_ofs & 1 ||
			(u32)usa_ofs + ((u32)usa_count * 2) > size ||
			(size >> NTFS_BLOCK_SIZE_SHIFT) != usa_count)
		return EINVAL;
	/* Position of usn in update sequence array. */
	usa_pos = (le16*)((u8*)b + usa_ofs);
	/*
	 * Cyclically increment the update sequence number (skipping 0 and -1,
	 * i.e. 0xffff).
	 */
	usn = le16_to_cpup(usa_pos) + 1;
	if (usn == 0xffff || !usn)
		usn = 1;
	le_usn = cpu_to_le16(usn);
	*usa_pos = le_usn;
	/* Position in data of first u16 that needs fixing up. */
	data_pos = (le16*)b + NTFS_BLOCK_SIZE/sizeof(le16) - 1;
	/* Fixup all sectors. */
	while (usa_count--) {
		/*
		 * Increment the position in the usa and save the
		 * original data from the data buffer into the usa.
		 */
		*(++usa_pos) = *data_pos;
		/* Apply fixup to data. */
		*data_pos = le_usn;
		/* Increment position in data as well. */
		data_pos += NTFS_BLOCK_SIZE/sizeof(le16);
	}
	return 0;
}

/**
 * ntfs_mst_fixup_post_write - fast deprotect mst protected data
 * @b:		pointer to the data to deprotect
 *
 * Perform the necessary post write multi sector transfer fixup, not checking
 * for any errors, because we assume we have just successfully used
 * ntfs_mst_fixup_pre_write(), thus the data will be fine or we would never
 * have gotten here.
 */
void ntfs_mst_fixup_post_write(NTFS_RECORD *b)
{
	le16 *usa_pos, *data_pos;
	u16 usa_ofs = le16_to_cpu(b->usa_ofs);
	u16 usa_count = le16_to_cpu(b->usa_count) - 1;
	/* Position of usn in update sequence array. */
	usa_pos = (le16*)b + usa_ofs/sizeof(le16);
	/* Position in protected data of first u16 that needs fixing up. */
	data_pos = (le16*)b + NTFS_BLOCK_SIZE/sizeof(le16) - 1;
	/* Fixup all sectors. */
	while (usa_count--) {
		/*
		 * Increment position in usa and restore original data from
		 * the usa into the data buffer.
		 */
		*data_pos = *(++usa_pos);
		/* Increment position in data as well. */
		data_pos += NTFS_BLOCK_SIZE/sizeof(le16);
	}
}
