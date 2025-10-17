/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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
#ifndef _SYS_XATTR_H_
#define _SYS_XATTR_H_

#include <sys/types.h>

/* Options for pathname based xattr calls */
#define XATTR_NOFOLLOW   0x0001     /* Don't follow symbolic links */

/* Options for setxattr calls */
#define XATTR_CREATE     0x0002     /* set the value, fail if attr already exists */
#define XATTR_REPLACE    0x0004     /* set the value, fail if attr does not exist */

/* Set this to bypass authorization checking (eg. if doing auth-related work) */
#define XATTR_NOSECURITY 0x0008

/* Set this to bypass the default extended attribute file (dot-underscore file) */
#define XATTR_NODEFAULT  0x0010

/* option for f/getxattr() and f/listxattr() to expose the HFS Compression extended attributes */
#define XATTR_SHOWCOMPRESSION 0x0020

/* Options for pathname based xattr calls */
#define XATTR_NOFOLLOW_ANY  0x0040  /* Don't follow any symbolic links in the path */

#define XATTR_MAXNAMELEN   127

/* See the ATTR_CMN_FNDRINFO section of getattrlist(2) for details on FinderInfo */
#define XATTR_FINDERINFO_NAME     "com.apple.FinderInfo"

#define XATTR_RESOURCEFORK_NAME   "com.apple.ResourceFork"


#ifdef KERNEL

#ifdef KERNEL_PRIVATE
#define XATTR_VNODE_SUPPORTED(vp) \
	((vp)->v_type == VREG || (vp)->v_type == VDIR || (vp)->v_type == VLNK || (vp)->v_type == VSOCK || (vp)->v_type == VFIFO)
#endif

__BEGIN_DECLS
int  xattr_protected(const char *);
int  xattr_validatename(const char *);

/* Maximum extended attribute size supported by VFS */
#define XATTR_MAXSIZE           INT32_MAX

#ifdef PRIVATE
/* Maximum extended attribute size in an Apple Double file */
#define AD_XATTR_MAXSIZE        XATTR_MAXSIZE

/* Number of bits used to represent the maximum size of
 * extended attribute stored in an Apple Double file.
 */
#define AD_XATTR_SIZE_BITS      31
#endif /* PRIVATE */

__END_DECLS
#endif /* KERNEL */

#ifndef KERNEL
__BEGIN_DECLS

ssize_t getxattr(const char *path, const char *name, void *value, size_t size, u_int32_t position, int options);

ssize_t fgetxattr(int fd, const char *name, void *value, size_t size, u_int32_t position, int options);

int setxattr(const char *path, const char *name, const void *value, size_t size, u_int32_t position, int options);

int fsetxattr(int fd, const char *name, const void *value, size_t size, u_int32_t position, int options);

int removexattr(const char *path, const char *name, int options);

int fremovexattr(int fd, const char *name, int options);

ssize_t listxattr(const char *path, char *namebuff, size_t size, int options);

ssize_t flistxattr(int fd, char *namebuff, size_t size, int options);

__END_DECLS
#endif /* KERNEL */

#endif /* _SYS_XATTR_H_ */
