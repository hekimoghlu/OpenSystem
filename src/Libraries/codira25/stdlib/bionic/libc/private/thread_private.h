/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
/* PUBLIC DOMAIN: No Rights Reserved. Marco S Hyman <marc@snafu.org> */

#pragma once

#include <pthread.h>

__BEGIN_DECLS

/*
 * This file defines the thread library interface to libc.  Thread
 * libraries must implement the functions described here for proper
 * inter-operation with libc.   libc contains weak versions of the
 * described functions for operation in a non-threaded environment.
 */

#define __MUTEX_NAME(name) __CONCAT(__libc_mutex_,name)
#define _THREAD_PRIVATE_MUTEX(name) static pthread_mutex_t __MUTEX_NAME(name) = PTHREAD_MUTEX_INITIALIZER
#define _THREAD_PRIVATE_MUTEX_LOCK(name) pthread_mutex_lock(&__MUTEX_NAME(name))
#define _THREAD_PRIVATE_MUTEX_UNLOCK(name) pthread_mutex_unlock(&__MUTEX_NAME(name))

/* Note that these aren't compatible with the usual OpenBSD ones which lazy-initialize! */
#define _MUTEX_LOCK(l) pthread_mutex_lock((pthread_mutex_t*) l)
#define _MUTEX_UNLOCK(l) pthread_mutex_unlock((pthread_mutex_t*) l)

__LIBC_HIDDEN__ void    _thread_arc4_lock(void);
__LIBC_HIDDEN__ void    _thread_arc4_unlock(void);

#define _ARC4_LOCK() _thread_arc4_lock()
#define _ARC4_UNLOCK() _thread_arc4_unlock()

extern volatile sig_atomic_t _rs_forked;

__END_DECLS
