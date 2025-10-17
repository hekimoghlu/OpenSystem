/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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

#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#include <stdio.h>

static int x = 0;
static int y = 0;

int main(void)
{
	/* found in <rdar://problem/16326400> 12A216: Spotlight takes a long time to show results */
	
	/* we need to start up NCPU-1 threads in a given bucket, then fire up one more at a separate
	 * priority.
	 *
	 * each of these waiters needs to be non-blocked until the point where dispatch wants to
	 * request a new thread.
	 *
	 * if dispatch ever fixes sync_barrier -> sync handoff to not require an extra thread,
	 * then this test will never fail and will be invalid.
	 */
	 
	printf("[TEST] barrier_sync -> async @ ncpu threads\n");
	 
	dispatch_semaphore_t sema = dispatch_semaphore_create(0);
	
	int ncpu = 1;
	size_t sz = sizeof(ncpu);
	sysctlbyname("hw.ncpu", &ncpu, &sz, NULL, 0);
	printf("starting up %d waiters.\n", ncpu);
	
	dispatch_queue_t q = dispatch_queue_create("moo", DISPATCH_QUEUE_CONCURRENT);
	dispatch_barrier_sync(q, ^{
		dispatch_async(q, ^{ 
			printf("async.\n"); 
			dispatch_semaphore_signal(sema);
		});
		for (int i=0; i<ncpu-1; i++) {
			dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
				printf("waiter %d* up.\n", i);
				while (y == 0) { };
			});
		}
		dispatch_async(dispatch_get_global_queue(0, 0), ^{
			printf("waiter %d up.\n", ncpu-1);
			while (x == 0) { };
			printf("waiter %d idle.\n", ncpu-1);
			usleep(1000);
			dispatch_sync(q, ^{ printf("quack %d\n", ncpu-1); });
		});
		printf("waiting...\n");
		sleep(1);
		printf("done.\n");
	});
	
	x = 1;
	int rv = dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, 2ull * NSEC_PER_SEC));
	printf("[%s] barrier_sync -> async completed\n", rv == 0 ? "PASS" : "FAIL");

	return rv;
}
