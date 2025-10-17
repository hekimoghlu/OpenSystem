/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
 * Copyright 1996 1995 by Open Software Foundation, Inc. 1997 1996 1995 1994 1993 1992 1991  
 *              All Rights Reserved 
 *  
 * Permission to use, copy, modify, and distribute this software and 
 * its documentation for any purpose and without fee is hereby granted, 
 * provided that the above copyright notice appears in all copies and 
 * that both the copyright notice and this permission notice appear in 
 * supporting documentation. 
 *  
 * OSF DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE 
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE. 
 *  
 * IN NO EVENT SHALL OSF BE LIABLE FOR ANY SPECIAL, INDIRECT, OR 
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM 
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN ACTION OF CONTRACT, 
 * NEGLIGENCE, OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION 
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. 
 * 
 */
/*
 * MkLinux
 */

/*
 * POSIX Threads - IEEE 1003.1c
 */

#ifndef _POSIX_PTHREAD_SPINLOCK_H
#define _POSIX_PTHREAD_SPINLOCK_H

#include <sys/cdefs.h>
#include <mach/mach.h>
#include <libkern/OSAtomic.h>

typedef volatile OSSpinLock pthread_lock_t __deprecated_msg("Use <os/lock.h> instead");

#define LOCK_INIT(l) ((l) = OS_SPINLOCK_INIT)
#define LOCK_INITIALIZER OS_SPINLOCK_INIT

#define _DO_SPINLOCK_LOCK(v) OSSpinLockLock(v)
#define _DO_SPINLOCK_UNLOCK(v) OSSpinLockUnlock(v)

#define TRY_LOCK(v) OSSpinLockTry((volatile OSSpinLock *)&(v))
#define LOCK(v) OSSpinLockLock((volatile OSSpinLock *)&(v))
#define UNLOCK(v) OSSpinLockUnlock((volatile OSSpinLock *)&(v))

extern void _spin_lock(pthread_lock_t *lockp) __deprecated_msg("Use <os/lock.h> instead");
extern int _spin_lock_try(pthread_lock_t *lockp) __deprecated_msg("Use <os/lock.h> instead");
extern void _spin_unlock(pthread_lock_t *lockp) __deprecated_msg("Use <os/lock.h> instead");

extern void spin_lock(pthread_lock_t *lockp) __deprecated_msg("Use <os/lock.h> instead");
extern int spin_lock_try(pthread_lock_t *lockp) __deprecated_msg("Use <os/lock.h> instead");
extern void spin_unlock(pthread_lock_t *lockp) __deprecated_msg("Use <os/lock.h> instead");

#endif /* _POSIX_PTHREAD_SPINLOCK_H */
