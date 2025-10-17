/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
/* Copyright (c) 1998 Apple Computer, Inc. All Rights Reserved */
/*-
 *	@(#)vstat.h
 */

#ifndef _SYS_VSTAT_H_
#define _SYS_VSTAT_H_

#include <sys/appleapiopts.h>
#include <sys/cdefs.h>
#include <sys/_types/_fsid_t.h>

#warning obsolete header! delete the include from your sources

#ifdef __APPLE_API_OBSOLETE

#include <sys/time.h>
#include <sys/attr.h>

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)

struct vstat {
	fsid_t                  vst_volid;              /* volume identifier */
	fsobj_id_t              vst_nodeid;             /* object's id */
	fsobj_type_t            vst_vnodetype;  /* vnode type (VREG, VDIR, etc.) */
	fsobj_tag_t             vst_vnodetag;   /* vnode tag (HFS, UFS, etc.) */
	mode_t                  vst_mode;               /* inode protection mode */
	nlink_t                 vst_nlink;              /* number of hard links */
	uid_t                   vst_uid;                /* user ID of the file's owner */
	gid_t                   vst_gid;                /* group ID of the file's group */
	dev_t                   vst_dev;                /* inode's device */
	dev_t                   vst_rdev;               /* device type */
#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
	struct  timespec vst_atimespec; /* time of last access */
	struct  timespec vst_mtimespec; /* time of last data modification */
	struct  timespec vst_ctimespec; /* time of last file status change */
#else
	time_t                  vst_atime;              /* time of last access */
	long                    vst_atimensec;  /* nsec of last access */
	time_t                  vst_mtime;              /* time of last data modification */
	long                    vst_mtimensec;  /* nsec of last data modification */
	time_t                  vst_ctime;              /* time of last file status change */
	long                    vst_ctimensec;  /* nsec of last file status change */
#endif
	off_t                   vst_filesize;   /* file size, in bytes */
	quad_t                  vst_blocks;             /* bytes allocated for file */
	u_int32_t               vst_blksize;    /* optimal blocksize for I/O */
	u_int32_t               vst_flags;              /* user defined flags for file */
};

#endif /* (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */
#endif /* __APPLE_API_OBSOLETE */

#endif /* !_SYS_VSTAT_H_ */
