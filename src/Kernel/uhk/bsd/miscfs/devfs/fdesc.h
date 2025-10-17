/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
/* Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved */
/*
 * Copyright (c) 1992, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software donated to Berkeley by
 * Jan-Simon Pendry.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)fdesc.h	8.6 (Berkeley) 8/20/94
 *
 * #Id: fdesc.h,v 1.8 1993/04/06 15:28:33 jsp Exp #
 */
#ifndef __FDESC_FDESC_H__
#define __FDESC_FDESC_H__

#include  <sys/appleapiopts.h>

#ifdef __APPLE_API_PRIVATE
#ifdef KERNEL

#define FD_ROOT         2
#define FD_DEVFD        3
#define FD_STDIN        4
#define FD_STDOUT       5
#define FD_STDERR       6
#define FD_DESC         8
#define FD_MAX          12

typedef enum {
	Fdesc,
} fdntype;

struct fdescnode {
	LIST_ENTRY(fdescnode)   fd_hash;        /* Hash list */
	struct vnode            *fd_vnode;      /* Back ptr to vnode */
	fdntype                 fd_type;        /* Type of this node */
	int                     fd_fd;          /* Fd to be dup'ed */
	const char              *fd_link;       /* Link to fd/n */
	int                     fd_ix;          /* filesystem index */
};

#define VFSTOFDESC(mp)  ((struct fdescmount *)((mp)->mnt_data))
#define VTOFDESC(vp) ((struct fdescnode *)(vp)->v_data)

__BEGIN_DECLS
extern int fdesc_allocvp(fdntype, int, struct mount *, struct vnode **, enum vtype, int);
extern int fdesc_badop(void);
extern int fdesc_getattr(struct vnop_getattr_args *ap);
extern int fdesc_inactive(struct vnop_inactive_args *ap);
extern int devfs_fdesc_init(void);
extern int devfs_fdesc_makelinks(void);
extern int fdesc_ioctl(struct vnop_ioctl_args *ap);
extern int devfs_devfd_lookup(struct vnop_lookup_args *ap);
extern int devfs_devfd_readdir(struct vnop_readdir_args *ap);
extern int fdesc_open(struct vnop_open_args *ap);
extern int fdesc_pathconf(struct vnop_pathconf_args *ap);
extern int fdesc_read(struct vnop_read_args *ap);
extern int fdesc_readdir(struct vnop_readdir_args *ap);
extern int fdesc_readlink(struct vnop_readlink_args *ap);
extern int fdesc_reclaim(struct vnop_reclaim_args *ap);
extern int fdesc_root(struct mount *, struct vnode **, vfs_context_t);
extern int fdesc_select(struct vnop_select_args *ap);
extern int fdesc_setattr(struct vnop_setattr_args *ap);
extern int fdesc_write(struct vnop_write_args *ap);

extern int(**fdesc_vnodeop_p)(void *);
extern int(**devfs_devfd_vnodeop_p)(void*);
extern struct vfsops fdesc_vfsops;
__END_DECLS

#endif /* KERNEL */
#endif /* __APPLE_API_PRIVATE */
#endif /* __FDESC_FDESC_H__ */
