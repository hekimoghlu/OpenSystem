/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
/* Copyright (c) 1995, 1997 Apple Computer, Inc. All Rights Reserved */
/*
 * Copyright (c) 1982, 1986, 1989, 1993
 *	The Regents of the University of California.  All rights reserved.
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
 *	@(#)file.h	8.3 (Berkeley) 1/9/95
 */

#ifndef _SYS_FILE_H_
#define _SYS_FILE_H_

#include <sys/appleapiopts.h>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <sys/unistd.h>
#include <sys/queue.h>
#include <sys/cdefs.h>

#ifdef KERNEL
#include <sys/queue.h>
#include <sys/kernel_types.h>
#include <sys/uio.h>
#endif

#ifndef _KAUTH_CRED_T
#define _KAUTH_CRED_T
struct ucred;
typedef struct ucred *kauth_cred_t;
struct posix_cred;
typedef struct posix_cred *posix_cred_t;
#endif  /* !_KAUTH_CRED_T */

__BEGIN_DECLS
#ifdef KERNEL
int file_socket(int, socket_t *);
int file_vnode(int, vnode_t *);
int file_vnode_withvid(int, vnode_t *, uint32_t *);
int file_flags(int, int *);
int file_drop(int);
#endif /* KERNEL */

#ifdef KERNEL_PRIVATE
struct fileglob;
struct fileproc;
struct vnode;
int fp_getfvp(struct proc *p, int fd, struct fileproc **resultfp, struct vnode  **resultvp);
int fp_get_pipe_id(proc_t p, int fd, uint64_t *result_pipe_id);
struct vnode *fg_get_vnode(struct fileglob *fg);
void generate_file_permissions_guard_exception(unsigned int code_target, int64_t subcode);
#define GEN_FILEPERM_EXCEPTION 1
#endif  /* KERNEL_PRIVATE */
__END_DECLS
#endif /* !_SYS_FILE_H_ */
