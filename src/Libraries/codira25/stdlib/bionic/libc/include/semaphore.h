/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#ifndef _SEMAPHORE_H
#define _SEMAPHORE_H

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

struct timespec;

typedef struct {
  unsigned int count;
#ifdef __LP64__
  int __reserved[3];
#endif
} sem_t;

#define SEM_FAILED __BIONIC_CAST(reinterpret_cast, sem_t*, 0)


#if __BIONIC_AVAILABILITY_GUARD(30)
int sem_clockwait(sem_t* _Nonnull __sem, clockid_t __clock, const struct timespec* _Nonnull __ts) __INTRODUCED_IN(30);
#endif /* __BIONIC_AVAILABILITY_GUARD(30) */

int sem_destroy(sem_t* _Nonnull __sem);
int sem_getvalue(sem_t* _Nonnull __sem, int* _Nonnull __value);
int sem_init(sem_t* _Nonnull __sem, int __shared, unsigned int __value);
int sem_post(sem_t* _Nonnull __sem);
int sem_timedwait(sem_t* _Nonnull __sem, const struct timespec* _Nonnull __ts);
/*
 * POSIX historically only supported using sem_timedwait() with CLOCK_REALTIME, however that is
 * typically inappropriate, since that clock can change dramatically, causing the timeout to either
 * expire earlier or much later than intended.  This function is added to use a timespec based
 * on CLOCK_MONOTONIC that does not suffer from this issue.
 * Note that sem_clockwait() allows specifying an arbitrary clock and has superseded this
 * function.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
int sem_timedwait_monotonic_np(sem_t* _Nonnull __sem, const struct timespec* _Nonnull __ts) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */

int sem_trywait(sem_t* _Nonnull __sem);
int sem_wait(sem_t* _Nonnull __sem);

/* These aren't actually implemented. */
sem_t* _Nullable sem_open(const char* _Nonnull __name, int _flags, ...);
int sem_close(sem_t* _Nonnull __sem);
int sem_unlink(const char* _Nonnull __name);

__END_DECLS

#endif
