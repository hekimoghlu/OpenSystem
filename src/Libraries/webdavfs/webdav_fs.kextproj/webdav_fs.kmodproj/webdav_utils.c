/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
#include "webdav_utils.h"

/*****************************************************************************/
/*
 * Lock a webdavnode
 */
__private_extern__ int webdav_lock(struct webdavnode *pt, enum webdavlocktype locktype)
{
	if (locktype == WEBDAV_SHARED_LOCK)
		lck_rw_lock_shared(&pt->pt_rwlock);
	else
		lck_rw_lock_exclusive(&pt->pt_rwlock);

	pt->pt_lockState = locktype;
	
#if 0
	/* For Debugging... */
	if (locktype != WEBDAV_SHARED_LOCK) {
		pt->pt_activation = (void *) current_thread();
	}
#endif

	return (0);
}

/*****************************************************************************/
/*
 * Unlock a webdavnode
 */
__private_extern__ void webdav_unlock(struct webdavnode *pt)
{
	lck_rw_done(&pt->pt_rwlock);
	pt->pt_lockState = 0;
}

void timespec_to_webdav_timespec64(struct timespec ts, struct webdav_timespec64 *wts)
{
	wts->tv_sec = ts.tv_sec;
	wts->tv_nsec = ts.tv_nsec;
}

void webdav_timespec64_to_timespec(struct webdav_timespec64 wts, struct timespec *ts)
{
#ifdef __LP64__
	ts->tv_sec = wts.tv_sec;
	ts->tv_nsec = wts.tv_nsec;
#else
	ts->tv_sec = (uint32_t)wts.tv_sec;
	ts->tv_nsec = (uint32_t)wts.tv_nsec;	
#endif
}