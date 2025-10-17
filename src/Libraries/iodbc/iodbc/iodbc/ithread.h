/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#ifndef _ITHREAD_H
#define _ITHREAD_H


/*
 *  Threading under windows
 */
#if defined (WIN32) && !defined (NO_THREADING)

# define IODBC_THREADING

# define THREAD_IDENT			((unsigned long) GetCurrentThreadId())

# define MUTEX_DECLARE(M)		HANDLE M
# define MUTEX_INIT(M)			M = CreateMutex (NULL, FALSE, NULL)
# define MUTEX_DONE(M)			CloseHandle (M)
# define MUTEX_LOCK(M)			WaitForSingleObject (M, INFINITE)
# define MUTEX_UNLOCK(M)		ReleaseMutex (M)

# define SPINLOCK_DECLARE(M)		CRITICAL_SECTION M
# define SPINLOCK_INIT(M)		InitializeCriticalSection (&M)
# define SPINLOCK_DONE(M)		DeleteCriticalSection (&M)
# define SPINLOCK_LOCK(M)		EnterCriticalSection (&M)
# define SPINLOCK_UNLOCK(M)		LeaveCriticalSection (&M)


/*
 *  Threading with pthreads
 */
#elif defined (WITH_PTHREADS)

#ifndef _REENTRANT
# error Add -D_REENTRANT to your compiler flags
#endif

#include <pthread.h>

# define IODBC_THREADING

# ifndef OLD_PTHREADS
#  define THREAD_IDENT			((unsigned long) (pthread_self ()))
# else
#  define THREAD_IDENT			0UL
# endif

# define MUTEX_DECLARE(M)		pthread_mutex_t M
# define MUTEX_INIT(M)			pthread_mutex_init (&M, NULL)
# define MUTEX_DONE(M)			pthread_mutex_destroy (&M)
# define MUTEX_LOCK(M)			pthread_mutex_lock(&M)
# define MUTEX_UNLOCK(M)		pthread_mutex_unlock(&M)

# define SPINLOCK_DECLARE(M)		MUTEX_DECLARE(M)
# define SPINLOCK_INIT(M)		MUTEX_INIT(M)
# define SPINLOCK_DONE(M)		MUTEX_DONE(M)
# define SPINLOCK_LOCK(M)		MUTEX_LOCK(M)
# define SPINLOCK_UNLOCK(M)		MUTEX_UNLOCK(M)


/*
 *  No threading
 */
#else
	
# undef IODBC_THREADING

# undef THREAD_IDENT

# define MUTEX_DECLARE(M)		int M
# define MUTEX_INIT(M)			M = 1
# define MUTEX_DONE(M)			M = 1
# define MUTEX_LOCK(M)			M = 1
# define MUTEX_UNLOCK(M)		M = 1

# define SPINLOCK_DECLARE(M)		MUTEX_DECLARE (M)
# define SPINLOCK_INIT(M)		MUTEX_INIT (M)
# define SPINLOCK_DONE(M)		MUTEX_DONE (M)
# define SPINLOCK_LOCK(M)		MUTEX_LOCK (M)
# define SPINLOCK_UNLOCK(M)		MUTEX_UNLOCK (M)

#endif

#endif /* _ITHREAD_H */
