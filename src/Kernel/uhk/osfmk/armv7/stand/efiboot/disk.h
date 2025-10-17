/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
#ifndef _DISK_H
#define _DISK_H

#include <sys/queue.h>

typedef struct efi_diskinfo {
	EFI_BLOCK_IO		*blkio;
	UINT32			 mediaid;
} *efi_diskinfo_t;

struct diskinfo {
	struct efi_diskinfo ed;
	struct disklabel disklabel;

	u_int part;
	u_int flags;
#define DISKINFO_FLAG_GOODLABEL		(1 << 0)

	int (*diskio)(int, struct diskinfo *, u_int, int, void *);
	int (*strategy)(void *, int, daddr_t, size_t, void *, size_t *);

	TAILQ_ENTRY(diskinfo) list;
};
TAILQ_HEAD(disklist_lh, diskinfo);

extern struct diskinfo *bootdev_dip;

extern struct disklist_lh disklist;

#endif /* _DISK_H */
