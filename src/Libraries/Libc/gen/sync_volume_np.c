/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#include <errno.h>
#include <unistd.h>
#include <System/sys/fsctl.h>

/*
 * Sync a mounted filesystem, without syncing others.
 * There are currently two flags that can be used:
 *
 * SYNC_VOLUME_FULLSYNC causes it to try to push the
 * data to the platter (otherwise, it just pushes it
 * to the disk drive, where it may remain in cache
 * for a while).
 *
 * SYNC_VOLUME_WAIT causes it to wait until the writes
 * have completed.  Otherwise, it'll return as soon
 * as the requests have been made.
 *
 * The functions are a simple wrapper for fsctl, and
 * return what it does.
 */

int
sync_volume_np(const char *path, int flags) {
	int full_sync = 0;
	int terrno;
	int rv;

	if (flags & SYNC_VOLUME_FULLSYNC)
		full_sync |= FSCTL_SYNC_FULLSYNC;

	if (flags & SYNC_VOLUME_WAIT)
		full_sync |= FSCTL_SYNC_WAIT;

	terrno = errno;
	rv = (fsctl(path, FSIOC_SYNC_VOLUME, &full_sync, 0) == -1) ? errno : 0;
	errno = terrno;
	return rv;
}

int
fsync_volume_np(int fd, int flags) {
	int full_sync = 0;
	int terrno;
	int rv;

	if (flags & SYNC_VOLUME_FULLSYNC)
		full_sync |= FSCTL_SYNC_FULLSYNC;

	if (flags & SYNC_VOLUME_WAIT)
		full_sync |= FSCTL_SYNC_WAIT;

	terrno = errno;
	rv = (ffsctl(fd, FSCTL_SYNC_VOLUME, &full_sync, 0) == -1) ? errno : 0;
	errno = terrno;
	return rv;
}

