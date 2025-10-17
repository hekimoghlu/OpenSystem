/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
/* $Id: rwlock.h,v 1.28 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_RWLOCK_H
#define ISC_RWLOCK_H 1

/*! \file isc/rwlock.h */

#include <isc/condition.h>
#include <isc/lang.h>
#include <isc/platform.h>
#include <isc/types.h>

ISC_LANG_BEGINDECLS

typedef enum {
	isc_rwlocktype_none = 0,
	isc_rwlocktype_read,
	isc_rwlocktype_write
} isc_rwlocktype_t;

#ifdef ISC_PLATFORM_USETHREADS
#if defined(ISC_PLATFORM_HAVEXADD) && defined(ISC_PLATFORM_HAVECMPXCHG)
#define ISC_RWLOCK_USEATOMIC 1
#endif

struct isc_rwlock {
	/* Unlocked. */
	unsigned int		magic;
	isc_mutex_t		lock;

#if defined(ISC_PLATFORM_HAVEXADD) && defined(ISC_PLATFORM_HAVECMPXCHG)
	/*
	 * When some atomic instructions with hardware assistance are
	 * available, rwlock will use those so that concurrent readers do not
	 * interfere with each other through mutex as long as no writers
	 * appear, massively reducing the lock overhead in the typical case.
	 *
	 * The basic algorithm of this approach is the "simple
	 * writer-preference lock" shown in the following URL:
	 * http://www.cs.rochester.edu/u/scott/synchronization/pseudocode/rw.html
	 * but our implementation does not rely on the spin lock unlike the
	 * original algorithm to be more portable as a user space application.
	 */

	/* Read or modified atomically. */
	isc_int32_t		write_requests;
	isc_int32_t		write_completions;
	isc_int32_t		cnt_and_flag;

	/* Locked by lock. */
	isc_condition_t		readable;
	isc_condition_t		writeable;
	unsigned int		readers_waiting;

	/* Locked by rwlock itself. */
	unsigned int		write_granted;

	/* Unlocked. */
	unsigned int		write_quota;

#else  /* ISC_PLATFORM_HAVEXADD && ISC_PLATFORM_HAVECMPXCHG */

	/*%< Locked by lock. */
	isc_condition_t		readable;
	isc_condition_t		writeable;
	isc_rwlocktype_t	type;

	/*% The number of threads that have the lock. */
	unsigned int		active;

	/*%
	 * The number of lock grants made since the lock was last switched
	 * from reading to writing or vice versa; used in determining
	 * when the quota is reached and it is time to switch.
	 */
	unsigned int		granted;
	
	unsigned int		readers_waiting;
	unsigned int		writers_waiting;
	unsigned int		read_quota;
	unsigned int		write_quota;
	isc_rwlocktype_t	original;
#endif  /* ISC_PLATFORM_HAVEXADD && ISC_PLATFORM_HAVECMPXCHG */
};
#else /* ISC_PLATFORM_USETHREADS */
struct isc_rwlock {
	unsigned int		magic;
	isc_rwlocktype_t	type;
	unsigned int		active;
};
#endif /* ISC_PLATFORM_USETHREADS */


isc_result_t
isc_rwlock_init(isc_rwlock_t *rwl, unsigned int read_quota,
		unsigned int write_quota);

isc_result_t
isc_rwlock_lock(isc_rwlock_t *rwl, isc_rwlocktype_t type);

isc_result_t
isc_rwlock_trylock(isc_rwlock_t *rwl, isc_rwlocktype_t type);

isc_result_t
isc_rwlock_unlock(isc_rwlock_t *rwl, isc_rwlocktype_t type);

isc_result_t
isc_rwlock_tryupgrade(isc_rwlock_t *rwl);

void
isc_rwlock_downgrade(isc_rwlock_t *rwl);

void
isc_rwlock_destroy(isc_rwlock_t *rwl);

ISC_LANG_ENDDECLS

#endif /* ISC_RWLOCK_H */
