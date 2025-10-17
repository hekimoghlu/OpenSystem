/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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
 * Copyright (c) 1991, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Scooter Morris at Genentech Inc.
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
 *	@(#)lockf.h	8.1 (Berkeley) 6/11/93
 * $FreeBSD: src/sys/sys/lockf.h,v 1.16 2004/04/07 04:19:49 imp Exp $
 */

#ifndef _SYS_LOCKF_H_
#define _SYS_LOCKF_H_

#include <sys/queue.h>
#include <sys/cdefs.h>
#include <sys/types.h>

struct vnop_advlock_args;
struct vnode;

#if IMPORTANCE_INHERITANCE
#define LF_NOT_BOOSTED  0
#define LF_BOOSTED      1
#endif /* IMPORTANCE_INHERITANCE */

/*
 * The lockf structure is a kernel structure which contains the information
 * associated with a byte range lock.  The lockf structures are linked into
 * the vnode structure.  Locks are sorted by the starting byte of the lock for
 * efficiency after they have been committed; uncommitted locks are on the list
 * head so they may quickly be accessed, and are both short lived and transient.
 */
TAILQ_HEAD(locklist, lockf);

struct lockf {
	short   lf_flags;           /* Semantics: F_POSIX, F_FLOCK, F_WAIT */
	short   lf_type;            /* Lock type: F_RDLCK, F_WRLCK */
#if IMPORTANCE_INHERITANCE
	int     lf_boosted;         /* Is the owner of the lock boosted */
#endif
	off_t   lf_start;           /* Byte # of the start of the lock */
	off_t   lf_end;             /* Byte # of the end of the lock (-1=EOF) */
	caddr_t lf_id;              /* Id of the resource holding the lock */
	struct  lockf **lf_head;    /* Back pointer to the head of the lockf list */
	struct  vnode *lf_vnode;    /* Back pointer to the inode */
	struct  lockf *lf_next;     /* Pointer to the next lock on this inode */
	struct  locklist lf_blkhd;  /* List of requests blocked on this lock */
	TAILQ_ENTRY(lockf) lf_block;/* A request waiting for a lock */
	struct  proc *lf_owner;     /* The proc that did the SETLK, if known */
};

__BEGIN_DECLS

#ifdef KERNEL_PRIVATE
void    lf_init(void);
int     lf_advlock(struct vnop_advlock_args *);
int     lf_assert(struct vnop_advlock_args *, void **);
void    lf_commit(void *, int);
void    lf_abort_advlocks(vnode_t);

#ifdef LOCKF_DEBUG
void    lf_print(char *, struct lockf *);
void    lf_printlist(char *, struct lockf *);
#endif
#endif /* KERNEL_PRIVATE */

__END_DECLS

#endif /* !_SYS_LOCKF_H_ */
