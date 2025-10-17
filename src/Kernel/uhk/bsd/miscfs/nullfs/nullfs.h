/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
/*-
 * Portions Copyright (c) 1992, 1993
 *  The Regents of the University of California.  All rights reserved.
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
 *  @(#)null.h  8.3 (Berkeley) 8/20/94
 *
 * $FreeBSD$
 */

#ifndef FS_NULL_H
#define FS_NULL_H

#include <sys/appleapiopts.h>
#include <libkern/libkern.h>
#include <sys/vnode.h>
#include <sys/vnode_if.h>
#include <sys/ubc.h>
#include <vfs/vfs_support.h>
#include <sys/lock.h>

#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/syslimits.h>

#if KERNEL
#include <libkern/tree.h>
#else
#include <System/libkern/tree.h>
#endif

// #define NULLFS_DEBUG 0

#define NULLM_CACHE 0x0001
#define NULLM_CASEINSENSITIVE 0x0000000000000002
#define NULLM_UNVEIL 0x1ULL << 2

typedef int (*vop_t)(void *);

struct null_mount {
	struct vnode * nullm_rootvp;       /* Reference to root null_node (inode 1) */
	struct vnode * nullm_secondvp;     /* Reference to virtual directory vnode to wrap app
	                                    *  bundles (inode 2) */
	struct vnode * nullm_lowerrootvp;  /* reference to the root of the tree we are
	                                    * relocating (in the other file system) */
	uint32_t nullm_lowerrootvid;       /* store the lower root vid so we can check
	                                    *  before we build the shadow vnode lazily*/
	lck_mtx_t nullm_lock;              /* lock to protect rootvp and secondvp above */
	uint64_t nullm_flags;
	uid_t uid;
	gid_t gid;
};

struct null_mount_conf {
	uint64_t flags;
};

#ifdef KERNEL

#define NULL_FLAG_HASHED 0x000000001

/*
 * A cache of vnode references
 */
struct null_node {
	LIST_ENTRY(null_node) null_hash; /* Hash list */
	struct vnode * null_lowervp;     /* VREFed once */
	struct vnode * null_vnode;       /* Back pointer */
	uint32_t null_lowervid;          /* vid for lowervp to detect lowervp getting recycled out
	                                  *  from under us */
	uint32_t null_myvid;
	uint32_t null_flags;
};

struct vnodeop_desc_fake {
	int vdesc_offset;
	const char * vdesc_name;
	/* other stuff */
};

#define NULLV_NOUNLOCK 0x0001
#define NULLV_DROP 0x0002

#define MOUNTTONULLMOUNT(mp) ((struct null_mount *)(vfs_fsprivate(mp)))
#define VTONULL(vp) ((struct null_node *)vnode_fsnode(vp))
#define NULLTOV(xp) ((xp)->null_vnode)

__BEGIN_DECLS

int nullfs_init(struct vfsconf * vfsp);
void nullfs_init_lck(lck_mtx_t * lck);
void nullfs_destroy_lck(lck_mtx_t * lck);
int nullfs_uninit(void);
int null_nodeget(
	struct mount * mp, struct vnode * lowervp, struct vnode * dvp, struct vnode ** vpp, struct componentname * cnp, int root);
int null_hashget(struct mount * mp, struct vnode * lowervp, struct vnode ** vpp);
int null_getnewvnode(
	struct mount * mp, struct vnode * lowervp, struct vnode * dvp, struct vnode ** vpp, struct componentname * cnp, int root);
void null_hashrem(struct null_node * xp);

int nullfs_getbackingvnode(vnode_t in_vp, vnode_t* out_vpp);

vfs_context_t nullfs_get_patched_context(struct null_mount * null_mp, vfs_context_t ctx);
void nullfs_cleanup_patched_context(struct null_mount * null_mp, vfs_context_t ctx);

#define NULLVPTOLOWERVP(vp) (VTONULL(vp)->null_lowervp)
#define NULLVPTOLOWERVID(vp) (VTONULL(vp)->null_lowervid)
#define NULLVPTOMYVID(vp) (VTONULL(vp)->null_myvid)

extern const struct vnodeopv_desc nullfs_vnodeop_opv_desc;

extern vop_t * nullfs_vnodeop_p;

__END_DECLS

#ifdef NULLFS_DEBUG
#define NULLFSDEBUG(format, args...) printf(format, ##args)
#else
#define NULLFSDEBUG(format, args...)
#endif /* NULLFS_DEBUG */

#endif /* KERNEL */

#endif
