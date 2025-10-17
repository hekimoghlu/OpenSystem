/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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

#include <string.h>

#include <kern/debug.h>

#include "ntfs_bitmap.h"
#include "ntfs_debug.h"
#include "ntfs_inode.h"
#include "ntfs_page.h"
#include "ntfs_types.h"
#include "ntfs_volume.h"

/**
 * __ntfs_bitmap_set_bits_in_run - set a run of bits in a bitmap to a value
 * @ni:			ntfs inode describing the bitmap
 * @start_bit:		first bit to set
 * @count:		number of bits to set
 * @value:		value to set the bits to (i.e. 0 or 1)
 * @is_rollback:	if true this is a rollback operation
 *
 * Set @count bits starting at bit @start_bit in the bitmap described by the
 * ntfs inode @ni to @value, where @value is either 0 or 1.
 *
 * @is_rollback should always be false, it is for internal use to rollback
 * errors.  You probably want to use ntfs_bitmap_set_bits_in_run() instead.
 *
 * Return 0 on success and errno on error.
 *
 * Locking: The caller must hold @ni->lock.
 */
errno_t __ntfs_bitmap_set_bits_in_run(ntfs_inode *ni, const s64 start_bit,
		const s64 count, const u8 value, const BOOL is_rollback)
{
	s64 rem, cnt = count;
	u64 ofs, end_ofs;
	upl_t upl;
	upl_page_info_array_t pl;
	u8 *kaddr;
	unsigned pos, len;
	errno_t err, err2;
	u8 bit;

	if (!ni)
		panic("%s(): !ni\n", __FUNCTION__);
	ntfs_debug("Entering for mft_no 0x%llx, start_bit 0x%llx, count "
			"0x%llx, value %u.%s", (unsigned long long)ni->mft_no,
			(unsigned long long)start_bit, (unsigned long long)cnt,
			(unsigned)value, is_rollback ? " (rollback)" : "");
	if (start_bit < 0)
		panic("%s(): start_bit < 0\n", __FUNCTION__);
	if (cnt < 0)
		panic("%s(): cnt < 0\n", __FUNCTION__);
	if (value > 1)
		panic("%s(): value > 1\n", __FUNCTION__);
	/*
	 * Calculate the offsets for the pages containing the first and last
	 * bits, i.e. @start_bit and @start_bit + @cnt - 1, respectively.
	 */
	ofs = (start_bit >> 3) & ~PAGE_MASK_64;
	end_ofs = ((start_bit + cnt - 1) >> 3) & ~PAGE_MASK_64;
	/* Get the page containing the first bit (@start_bit). */
	err = ntfs_page_map(ni, ofs, &upl, &pl, &kaddr, TRUE);
	if (err) {
		if (!is_rollback)
			ntfs_error(ni->vol->mp, "Failed to map first page "
					"(error %d), aborting.", err);
		return err;
	}
	/* Set @pos to the position of the byte containing @start_bit. */
	pos = (start_bit >> 3) & PAGE_MASK;
	/* Calculate the position of @start_bit in the first byte. */
	bit = start_bit & 7;
	/* If the first byte is partial, modify the appropriate bits in it. */
	if (bit) {
		u8 *byte = kaddr + pos;
		while ((bit & 7) && cnt) {
			cnt--;
			if (value)
				*byte |= 1 << bit++;
			else
				*byte &= ~(1 << bit++);
		}
		/* If we are done, unmap the page and return success. */
		if (!cnt)
			goto done;
		/* Update @pos to the new position. */
		pos++;
	}
	/*
	 * Depending on @value, modify all remaining whole bytes in the page up
	 * to @cnt.
	 */
	len = PAGE_SIZE - pos;
	rem = cnt >> 3;
	if (len > rem)
		len = rem;
	memset(kaddr + pos, value ? 0xff : 0, len);
	cnt -= len << 3;
	/* Update @len to point to the first not-done byte in the page. */
	if (cnt < 8)
		len += pos;
	/* If we are not in the last page, deal with all subsequent pages. */
	while (ofs < end_ofs) {
		if (cnt <= 0)
			panic("%s(): cnt <= 0\n", __FUNCTION__);
		/* Mark the current page dirty and unmap it. */
		ntfs_page_unmap(ni, upl, pl, TRUE);
		/* Update @ofs and get the next page. */
		ofs += PAGE_SIZE;
		err = ntfs_page_map(ni, ofs, &upl, &pl, &kaddr, TRUE);
		if (err)
			goto rollback;
		/*
		 * Depending on @value, modify all remaining whole bytes in the
		 * page up to @cnt.
		 */
		len = PAGE_SIZE;
		rem = cnt >> 3;
		if (len > rem)
			len = rem;
		memset(kaddr, value ? 0xff : 0, len);
		cnt -= len << 3;
	}
	/*
	 * The currently mapped page is the last one.  If the last byte is
	 * partial, modify the appropriate bits in it.  Note, @len is the
	 * position of the last byte inside the page.
	 */
	if (cnt) {
		u8 *byte;

		if (cnt > 7)
			panic("%s(): cnt > 7\n", __FUNCTION__);
		bit = cnt;
		byte = kaddr + len;
		while (bit--) {
			if (value)
				*byte |= 1 << bit;
			else
				*byte &= ~(1 << bit);
		}
	}
done:
	/* We are done.  Unmap the page and return success. */
	ntfs_page_unmap(ni, upl, pl, TRUE);
	ntfs_debug("Done.");
	return 0;
rollback:
	/*
	 * Current state:
	 *	- no pages are mapped
	 *	- @count - @cnt is the number of bits that have been modified
	 */
	if (is_rollback)
		return err;
	err2 = 0;
	if (count != cnt)
		err2 = __ntfs_bitmap_set_bits_in_run(ni, start_bit,
				count - cnt, value ? 0 : 1, TRUE);
	if (!err2) {
		/* Rollback was successful. */
		ntfs_error(ni->vol->mp, "Failed to map subsequent page (error "
				"%d), aborting.", err);
	} else {
		/* Rollback failed. */
		ntfs_error(ni->vol->mp, "Failed to map subsequent page (error "
				"%d) and rollback failed (error %d).  "
				"Aborting and leaving inconsistent metadata.  "
				"Unmount and run chkdsk.", err, err2);
		NVolSetErrors(ni->vol);
	}
	return err;
}
