/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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

/*
 * This file implements a barrier, a synchronization primitive designed to allow
 * threads to wait for each other at given points.  Barriers are initialized
 * with a given number of threads, n, using barrier_init().  When a thread calls
 * barrier_wait(), that thread blocks until n - 1 other threads reach the
 * barrier_wait() call using the same barrier_t.  When n threads have reached
 * the barrier, they are all awakened and sent on their way.  One of the threads
 * returns from barrier_wait() with a return code of 1; the remaining threads
 * get a return code of 0.
 */
#include <pthread.h>
#include <stdio.h>

#include "barrier.h"

void
barrier_init(barrier_t *bar, int nthreads)
{
	pthread_mutex_init(&bar->bar_lock, NULL);
	bar->bar_sem = dispatch_semaphore_create(0);

	bar->bar_numin = 0;
	bar->bar_nthr = nthreads;
}

int
barrier_wait(barrier_t *bar)
{
	pthread_mutex_lock(&bar->bar_lock);

	if (++bar->bar_numin < bar->bar_nthr) {
		pthread_mutex_unlock(&bar->bar_lock);
		dispatch_semaphore_wait(bar->bar_sem, DISPATCH_TIME_FOREVER);

		return (0);

	} else {
		int i;

		/* reset for next use */
		bar->bar_numin = 0;
		for (i = 1; i < bar->bar_nthr; i++)
			dispatch_semaphore_signal(bar->bar_sem);
		pthread_mutex_unlock(&bar->bar_lock);

		return (1);
	}
}

