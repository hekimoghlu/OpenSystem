/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
/*	@(#)semaphore.h	1.0	2/29/00		*/



/*
 * semaphore.h - POSIX semaphores
 *
 * HISTORY
 * 29-Feb-00	A.Ramesh at Apple
 *	Created for Mac OS X
 */

#ifndef _SYS_SEMAPHORE_H_
#define _SYS_SEMAPHORE_H_

typedef int sem_t;

/* this should go in limits.h> */
#define SEM_VALUE_MAX 32767
#define SEM_FAILED ((sem_t *)-1)

#ifndef KERNEL
#include <sys/cdefs.h>

__BEGIN_DECLS
int sem_close(sem_t *);
int sem_destroy(sem_t *) __deprecated;
int sem_getvalue(sem_t * __restrict, int * __restrict) __deprecated;
int sem_init(sem_t *, int, unsigned int) __deprecated;
sem_t * sem_open(const char *, int, ...);
int sem_post(sem_t *);
int sem_trywait(sem_t *);
int sem_unlink(const char *);
int sem_wait(sem_t *) __DARWIN_ALIAS_C(sem_wait);
__END_DECLS

#else   /* KERNEL */
void psem_cache_init(void);
#endif  /* KERNEL */

#endif  /* _SYS_SEMAPHORE_H_ */
