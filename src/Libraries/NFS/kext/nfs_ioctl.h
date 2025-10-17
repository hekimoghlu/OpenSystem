/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
 * Header file to export nfs defined ioctls for nfs_vnop_ioctls
 */

#ifndef _NFS_NFS_IOCTL_H_
#define _NFS_NFS_IOCTL_H_
#include <sys/ioccom.h>

/*
 * fsctl (vnop_ioctl) to detroy the callers credentials associated with the vnode's mount
 */
#define NFS_IOC_DESTROY_CRED            _IO('n', 1)

/*
 * fsctl (vnop_ioctl) to set the callers credentials associated with the vnode's mount
 */
struct nfs_gss_principal {
	size_t          princlen;       /* length of data */
	uint32_t        nametype;       /* nametype of data */
#ifdef KERNEL
	user32_addr_t   principal;      /* principal data in userspace */
#else
	uint8_t         *principal;
#endif
	uint32_t        flags;          /* Return flags */
};

#ifdef KERNEL
/* LP64 version of nfs_gss_principal */
struct user_nfs_gss_principal {
	size_t          princlen;       /* length of data */
	uint32_t        nametype;       /* nametype of data */
	user64_addr_t   principal;      /* principal data in userspace */
	uint32_t        flags;          /* Returned flags */
};
#endif

/* If no credential was found returned NFS_IOC_NO_CRED_FLAG in the flags field. */
#define NFS_IOC_NO_CRED_FLAG            1       /* No credential was found */
#define NFS_IOC_INVALID_CRED_FLAG       2       /* Found a credential, but its not valid */

#define NFS_IOC_SET_CRED                _IOW('n', 2, struct nfs_gss_principal)

#define NFS_IOC_GET_CRED                _IOWR('n', 3, struct nfs_gss_principal)

#define NFS_IOC_DISARM_TRIGGER          _IO('n', 4)

#ifdef KERNEL

#define NFS_IOC_SET_CRED64              _IOW('n', 2, struct user_nfs_gss_principal)

#define NFS_IOC_GET_CRED64              _IOWR('n', 3, struct user_nfs_gss_principal)
#endif

#endif
