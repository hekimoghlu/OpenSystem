/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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
/*
 * Copyright (c) 1997-2002 Apple Inc. All Rights Reserved
 *
 */

#ifndef _HFS_MOUNT_H_
#define _HFS_MOUNT_H_

#include <sys/appleapiopts.h>

#include <sys/mount.h>
#include <sys/time.h>

/*
 * Arguments to mount HFS-based filesystems
 */

#define OVERRIDE_UNKNOWN_PERMISSIONS 0

#define UNKNOWNUID ((uid_t)99)
#define UNKNOWNGID ((gid_t)99)
#define UNKNOWNPERMISSIONS (S_IRWXU | S_IROTH | S_IXOTH)		/* 705 */

#ifdef __APPLE_API_UNSTABLE
struct hfs_mount_args {
#ifndef KERNEL
	char	*fspec;			/* block special device to mount */
#endif
	uid_t	hfs_uid;		/* uid that owns hfs files (standard HFS only) */
	gid_t	hfs_gid;		/* gid that owns hfs files (standard HFS only) */
	mode_t	hfs_mask;		/* mask to be applied for hfs perms  (standard HFS only) */
	u_int32_t hfs_encoding;	/* encoding for this volume (standard HFS only) */
	struct	timezone hfs_timezone;	/* user time zone info (standard HFS only) */
	int		flags;			/* mounting flags, see below */
	int     journal_tbuffer_size;   /* size in bytes of the journal transaction buffer */
	int		journal_flags;          /* flags to pass to journal_open/create */
	int		journal_disable;        /* don't use journaling (potentially dangerous) */
};

#define HFSFSMNT_NOXONFILES	0x1	/* disable execute permissions for files */
#define HFSFSMNT_WRAPPER	0x2	/* mount HFS wrapper (if it exists) */
#define HFSFSMNT_EXTENDED_ARGS  0x4     /* indicates new fields after "flags" are valid */

/*
 * Sysctl values for HFS
 */
#define HFS_ENCODINGBIAS	1	    /* encoding matching CJK bias */
#define HFS_EXTEND_FS		2
#define HFS_ENABLE_JOURNALING   0x082969
#define HFS_DISABLE_JOURNALING  0x031272
#define HFS_REPLAY_JOURNAL	0x6a6e6c72
#define HFS_ENABLE_RESIZE_DEBUG 4	/* enable debug code for volume resizing */

#endif /* __APPLE_API_UNSTABLE */

#endif /* ! _HFS_MOUNT_H_ */
