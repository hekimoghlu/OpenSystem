/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
 * sys/statvfs.h
 */
#ifndef _SYS_STATVFS_H_
#define	_SYS_STATVFS_H_

#include <_bounds.h>
#include <sys/_types.h>
#include <sys/cdefs.h>

#include <sys/_types/_fsblkcnt_t.h>
#include <sys/_types/_fsfilcnt_t.h>

_LIBC_SINGLE_BY_DEFAULT()

/* Following structure is used as a statvfs/fstatvfs function parameter */
struct statvfs {
	unsigned long	f_bsize;	/* File system block size */
	unsigned long	f_frsize;	/* Fundamental file system block size */
	fsblkcnt_t	f_blocks;	/* Blocks on FS in units of f_frsize */
	fsblkcnt_t	f_bfree;	/* Free blocks */
	fsblkcnt_t	f_bavail;	/* Blocks available to non-root */
	fsfilcnt_t	f_files;	/* Total inodes */
	fsfilcnt_t	f_ffree;	/* Free inodes */
	fsfilcnt_t	f_favail;	/* Free inodes for non-root */
	unsigned long	f_fsid;		/* Filesystem ID */
	unsigned long	f_flag;		/* Bit mask of values */
	unsigned long	f_namemax;	/* Max file name length */
};

/* Defined bits for f_flag field value */
#define	ST_RDONLY	0x00000001	/* Read-only file system */
#define	ST_NOSUID	0x00000002	/* Does not honor setuid/setgid */

__BEGIN_DECLS
int fstatvfs(int, struct statvfs *);
int statvfs(const char * __restrict, struct statvfs * __restrict);
__END_DECLS

#endif	/* _SYS_STATVFS_H_ */
