/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

#include <kern/locks.h>

#include "ntfs_debug.h"
#include "ntfs_endian.h"
#include "ntfs_page.h"
#include "ntfs_time.h"
#include "ntfs_types.h"
#include "ntfs_usnjrnl.h"
#include "ntfs_volume.h"

/**
 * ntfs_usnjrnl_stamp - stamp the transaction log ($UsnJrnl) on an ntfs volume
 * @vol:	ntfs volume on which to stamp the transaction log
 *
 * Stamp the transaction log ($UsnJrnl) on the ntfs volume @vol and return 0
 * on success and errno on error.
 *
 * This function assumes that the transaction log has already been loaded and
 * consistency checked by a call to ntfs_vfsops.c::ntfs_usnjrnl_load().
 */
errno_t ntfs_usnjrnl_stamp(ntfs_volume *vol)
{
	ntfs_debug("Entering.");
	if (!NVolUsnJrnlStamped(vol)) {
		sle64 j_size, stamp;
		upl_t upl;
		upl_page_info_array_t pl;
		USN_HEADER *uh;
		ntfs_inode *max_ni;
		errno_t err;

		lck_spin_lock(&vol->usnjrnl_j_ni->size_lock);
		j_size = vol->usnjrnl_j_ni->data_size;
		lck_spin_unlock(&vol->usnjrnl_j_ni->size_lock);
		max_ni = vol->usnjrnl_max_ni;
		err = vnode_get(max_ni->vn);
		if (err) {
			ntfs_error(vol->mp, "Failed to get vnode for "
					"$UsnJrnl/$DATA/$Max.");
			return err;
		}
		lck_rw_lock_shared(&max_ni->lock);
		err = ntfs_page_map(max_ni, 0, &upl, &pl, (u8**)&uh, TRUE);
		if (err) {
			ntfs_error(vol->mp, "Failed to read from "
					"$UsnJrnl/$DATA/$Max attribute.");
			(void)vnode_put(max_ni->vn);
			return err;
		}
		stamp = ntfs_current_time();
		ntfs_debug("Stamping transaction log ($UsnJrnl): old "
				"journal_id 0x%llx, old lowest_valid_usn "
				"0x%llx, new journal_id 0x%llx, new "
				"lowest_valid_usn 0x%llx.",
				(unsigned long long)
				sle64_to_cpu(uh->journal_id),
				(unsigned long long)
				sle64_to_cpu(uh->lowest_valid_usn),
				(unsigned long long)sle64_to_cpu(stamp),
				(unsigned long long)j_size);
		uh->lowest_valid_usn = cpu_to_sle64(j_size);
		uh->journal_id = stamp;
		ntfs_page_unmap(max_ni, upl, pl, TRUE);
		lck_rw_unlock_shared(&max_ni->lock);
		(void)vnode_put(max_ni->vn);
		/* Set the flag so we do not have to do it again on remount. */
		NVolSetUsnJrnlStamped(vol);
		// TODO: Should we mark any times on the base inode $UsnJrnl
		// for update here?
	}
	ntfs_debug("Done.");
	return 0;
}
