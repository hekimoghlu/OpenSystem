/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
 * Copyright 2002 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _BARRIER_H
#define	_BARRIER_H

#include <dispatch/dispatch.h>
#include <pthread.h>

typedef struct barrier {
	pthread_mutex_t bar_lock;	/* protects bar_numin */
	int bar_numin;			/* current number of waiters */

	dispatch_semaphore_t *bar_sem;	/* where everyone waits */
	int bar_nthr;			/* # of waiters to trigger release */
} barrier_t;

extern void barrier_init(barrier_t *, int);
extern int barrier_wait(barrier_t *);

#endif /* _BARRIER_H */
