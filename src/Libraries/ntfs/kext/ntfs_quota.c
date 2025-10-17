/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#include <sys/stat.h>
#include <sys/ucred.h>
#include <sys/vnode.h>

#include "ntfs_debug.h"
#include "ntfs_endian.h"
#include "ntfs_index.h"
#include "ntfs_inode.h"
#include "ntfs_layout.h"
#include "ntfs_quota.h"
#include "ntfs_time.h"
#include "ntfs_types.h"
#include "ntfs_volume.h"

/**
 * ntfs_quotas_mark_out_of_date - mark the quotas out of date on an ntfs volume
 * @vol:	ntfs volume on which to mark the quotas out of date
 *
 * Mark the quotas out of date on the ntfs volume @vol and return 0 on success
 * and errno on error.
 */
errno_t ntfs_quotas_mark_out_of_date(ntfs_volume *vol)
{
	ntfs_inode *quota_ni;
	ntfs_index_context *ictx;
	INDEX_ENTRY *ie;
	QUOTA_CONTROL_ENTRY *qce;
	const le32 qid = QUOTA_DEFAULTS_ID;
	errno_t err;

	ntfs_debug("Entering.");
	if (NVolQuotaOutOfDate(vol))
		goto done;
	quota_ni = vol->quota_ni;
	if (!quota_ni || !vol->quota_q_ni) {
		ntfs_error(vol->mp, "Quota inodes are not open.");
		return EINVAL;
	}
	err = vnode_get(vol->quota_q_ni->vn);
	if (err) {
		ntfs_error(vol->mp, "Failed to get index vnode for "
				"$Quota/$Q.");
		return err;
	}
	lck_rw_lock_exclusive(&vol->quota_q_ni->lock);
	ictx = ntfs_index_ctx_get(vol->quota_q_ni);
	if (!ictx) {
		ntfs_error(vol->mp, "Failed to get index context.");
		err = ENOMEM;
		goto err;
	}
	err = ntfs_index_lookup(&qid, sizeof(qid), &ictx);
	if (err) {
		if (err == ENOENT)
			ntfs_error(vol->mp, "Quota defaults entry is not "
					"present.");
		else
			ntfs_error(vol->mp, "Lookup of quota defaults entry "
					"failed.");
		goto err;
	}
	ie = ictx->entry;
	if (le16_to_cpu(ie->data_length) <
			offsetof(QUOTA_CONTROL_ENTRY, sid)) {
		ntfs_error(vol->mp, "Quota defaults entry size is invalid.  "
				"Run chkdsk.");
		err = EIO;
		goto err;
	}
	qce = (QUOTA_CONTROL_ENTRY*)((u8*)ie + le16_to_cpu(ie->data_offset));
	if (le32_to_cpu(qce->version) != QUOTA_VERSION) {
		ntfs_error(vol->mp, "Quota defaults entry version 0x%x is not "
				"supported.", le32_to_cpu(qce->version));
		err = EIO;
		goto err;
	}
	ntfs_debug("Quota defaults flags = 0x%x.", le32_to_cpu(qce->flags));
	/* If quotas are already marked out of date, no need to do anything. */
	if (qce->flags & QUOTA_FLAG_OUT_OF_DATE)
		goto set_done;
	/*
	 * If quota tracking is neither requested nor enabled and there are no
	 * pending deletes, no need to mark the quotas out of date.
	 */
	if (!(qce->flags & (QUOTA_FLAG_TRACKING_ENABLED |
			QUOTA_FLAG_TRACKING_REQUESTED |
			QUOTA_FLAG_PENDING_DELETES)))
		goto set_done;
	/*
	 * Set the QUOTA_FLAG_OUT_OF_DATE bit thus marking quotas out of date.
	 * This is verified on WinXP to be sufficient to cause windows to
	 * rescan the volume on boot and update all quota entries.
	 */
	qce->flags |= QUOTA_FLAG_OUT_OF_DATE;
	/* Ensure the modified flags are written to disk. */
	ntfs_index_entry_mark_dirty(ictx);
	/* Update the atime, mtime and ctime of the base inode @quota_ni. */
	quota_ni->last_access_time = quota_ni->last_mft_change_time =
			quota_ni->last_data_change_time =
			ntfs_utc_current_time();
	NInoSetDirtyTimes(quota_ni);
set_done:
	ntfs_index_ctx_put(ictx);
	lck_rw_unlock_exclusive(&vol->quota_q_ni->lock);
	(void)vnode_put(vol->quota_q_ni->vn);
	/*
	 * We set the flag so we do not try to mark the quotas out of date
	 * again on remount.
	 */
	NVolSetQuotaOutOfDate(vol);
done:
	ntfs_debug("Done.");
	return 0;
err:
	if (ictx)
		ntfs_index_ctx_put(ictx);
	lck_rw_unlock_exclusive(&vol->quota_q_ni->lock);
	(void)vnode_put(vol->quota_q_ni->vn);
	return err;
}
