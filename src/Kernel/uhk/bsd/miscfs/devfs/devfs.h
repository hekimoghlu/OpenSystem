/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
 * Copyright 1997,1998 Julian Elischer.  All rights reserved.
 * julian@freebsd.org
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * miscfs/devfs/devfs.h
 */

#ifndef _MISCFS_DEVFS_DEVFS_H_
#define _MISCFS_DEVFS_DEVFS_H_

#include <sys/appleapiopts.h>
#include <sys/cdefs.h>

#define DEVFS_CHAR      0
#define DEVFS_BLOCK     1

/*
 * Argument to clone callback after dev
 */
#define DEVFS_CLONE_ALLOC       1       /* Allocate minor number slot */
#define DEVFS_CLONE_FREE        0       /* Free minor number slot */

__BEGIN_DECLS

/*
 * Function: devfs_make_node_clone
 *
 * Purpose
 *   Create a device node with the given pathname in the devfs namespace;
 *   before returning a dev_t value for an open instance, the dev_t has
 *   it's minor number updated by calling the supplied clone function on
 *   the supplied dev..
 *
 * Parameters:
 *   dev        - the dev_t value to associate
 *   chrblk	- block or character device (DEVFS_CHAR or DEVFS_BLOCK)
 *   uid, gid	- ownership
 *   perms	- permissions
 *   clone	- minor number cloning function
 *   fmt, ...	- print format string and args to format the path name
 * Returns:
 *   A handle to a device node if successful, NULL otherwise.
 */
void *  devfs_make_node_clone(dev_t dev, int chrblk, uid_t uid, gid_t gid,
    int perms, int (*clone)(dev_t dev, int action),
    const char *fmt, ...) __printflike(7, 8);

/*
 * Function: devfs_make_node
 *
 * Purpose
 *   Create a device node with the given pathname in the devfs namespace.
 *
 * Parameters:
 *   dev        - the dev_t value to associate
 *   chrblk	- block or character device (DEVFS_CHAR or DEVFS_BLOCK)
 *   uid, gid	- ownership
 *   perms	- permissions
 *   fmt, ...	- print format string and args to format the path name
 * Returns:
 *   A handle to a device node if successful, NULL otherwise.
 */
void *  devfs_make_node(dev_t dev, int chrblk, uid_t uid, gid_t gid,
    int perms, const char *fmt, ...) __printflike(6, 7);

#ifdef BSD_KERNEL_PRIVATE
/*
 * Function: devfs_make_link
 *
 * Purpose:
 *   Create a link to a previously created device node.
 *
 * Returns:
 *   0 if successful, -1 if failed
 */
int     devfs_make_link(void * handle, char *fmt, ...) __printflike(2, 3);
#endif /* BSD_KERNEL_PRIVATE */

/*
 * Function: devfs_remove
 *
 * Purpose:
 *   Remove the device node returned by devfs_make_node() along with
 *   any links created with devfs_make_link().
 */
void    devfs_remove(void * handle);

__END_DECLS

#ifdef __APPLE_API_PRIVATE
/* XXX */
#define UID_ROOT        0
#define UID_BIN         3
#define UID_UUCP        66
#define UID_LOGD        272

/* XXX */
#define GID_WHEEL       0
#define GID_KMEM        2
#define GID_TTY         4
#define GID_OPERATOR    5
#define GID_BIN         7
#define GID_GAMES       13
#define GID_DIALER      68
#define GID_WINDOWSERVER 88
#define GID_LOGD        272
#endif /* __APPLE_API_PRIVATE */

#endif /* !_MISCFS_DEVFS_DEVFS_H_ */
