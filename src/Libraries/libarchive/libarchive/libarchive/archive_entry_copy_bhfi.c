/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#include "archive_platform.h"

#include "archive_private.h"
#include "archive_entry.h"

#if defined(_WIN32) && !defined(__CYGWIN__)

#define EPOC_TIME ARCHIVE_LITERAL_ULL(116444736000000000)

__inline static void
fileTimeToUtc(const FILETIME *filetime, time_t *t, long *ns)
{
	ULARGE_INTEGER utc;

	utc.HighPart = filetime->dwHighDateTime;
	utc.LowPart  = filetime->dwLowDateTime;
	if (utc.QuadPart >= EPOC_TIME) {
		utc.QuadPart -= EPOC_TIME;
		*t = (time_t)(utc.QuadPart / 10000000);	/* milli seconds base */
		*ns = (long)(utc.QuadPart % 10000000) * 100;/* nano seconds base */
	} else {
		*t = 0;
		*ns = 0;
	}
}

void
archive_entry_copy_bhfi(struct archive_entry *entry,
			BY_HANDLE_FILE_INFORMATION *bhfi)
{
	time_t secs;
	long nsecs;

	fileTimeToUtc(&bhfi->ftLastAccessTime, &secs, &nsecs);
	archive_entry_set_atime(entry, secs, nsecs);
	fileTimeToUtc(&bhfi->ftLastWriteTime, &secs, &nsecs);
	archive_entry_set_mtime(entry, secs, nsecs);
	fileTimeToUtc(&bhfi->ftCreationTime, &secs, &nsecs);
	archive_entry_set_birthtime(entry, secs, nsecs);
	archive_entry_set_ctime(entry, secs, nsecs);
	archive_entry_set_dev(entry, bhfi->dwVolumeSerialNumber);
	archive_entry_set_ino64(entry, (((int64_t)bhfi->nFileIndexHigh) << 32)
		+ bhfi->nFileIndexLow);
	archive_entry_set_nlink(entry, bhfi->nNumberOfLinks);
	archive_entry_set_size(entry, (((int64_t)bhfi->nFileSizeHigh) << 32)
		+ bhfi->nFileSizeLow);
	/* archive_entry_set_mode(entry, st->st_mode); */
}
#endif
