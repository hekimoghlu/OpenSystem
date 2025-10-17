/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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

#include <darwintest.h>
#include <dispatch/dispatch.h>
#include <malloc/malloc.h>
#include <os/lock.h>
#include <stdlib.h>
#include <sys/queue.h>

#if TARGET_OS_WATCH
#define TEST_TIMEOUT 1200
#endif // TARGET_OS_WATCH

TAILQ_HEAD(thead, entry);
struct entry {
	TAILQ_ENTRY(entry) next;
};

static void
stress(size_t sz, size_t cnt)
{
	struct thead head = TAILQ_HEAD_INITIALIZER(head);
	TAILQ_INIT(&head);

	for (int t=0; t<100; t++) {
		for (int i=0; i<cnt; i++) {
			struct entry *p = calloc(1, sz);
			TAILQ_INSERT_TAIL(&head, p, next);
		}
		int i=0;
		struct entry *p;
		while ((p = TAILQ_FIRST(&head)) != NULL) {
			TAILQ_REMOVE(&head, p, next);
			free((void *)p);
			i++;
		}
	}
}

T_DECL(tiny_mem_pressure_multi, "test memory pressure in tiny on threads",
#if TARGET_OS_WATCH
		T_META_TIMEOUT(TEST_TIMEOUT),
#endif // TARGET_OS_WATCH
		T_META_CHECK_LEAKS(false)) {
	dispatch_group_t g = dispatch_group_create();
	for (int i=0; i<16; i++) {
		dispatch_group_async(g, dispatch_get_global_queue(0, 0), ^{
			stress(128, 100000);
		});
	}
	dispatch_group_notify(g, dispatch_get_global_queue(0, 0), ^{
		T_PASS("didn't crash!");
		T_END;
	});
	dispatch_release(g);

	while (1) {
		T_LOG("malloc_zone_pressure_relief");
		malloc_zone_pressure_relief(malloc_default_zone(), 0);
		sleep(1);
	}
}

T_DECL(small_mem_pressure_multi, "test memory pressure in small on threads",
#if TARGET_OS_WATCH
		T_META_TIMEOUT(TEST_TIMEOUT),
#endif // TARGET_OS_WATCH
		T_META_CHECK_LEAKS(false)) {
	dispatch_group_t g = dispatch_group_create();
	for (int i=0; i<3; i++) {
		dispatch_group_async(g, dispatch_get_global_queue(0, 0), ^{
			stress(1024, 100000);
		});
	}
	dispatch_group_notify(g, dispatch_get_global_queue(0, 0), ^{
		T_PASS("didn't crash!");
		T_END;
	});
	dispatch_release(g);

	while (1) {
		T_LOG("malloc_zone_pressure_relief");
		malloc_zone_pressure_relief(malloc_default_zone(), 0);
		sleep(1);
	}
}

T_DECL(medium_mem_pressure_multi, "test memory pressure in medium on threads",
#if TARGET_OS_WATCH
		T_META_TIMEOUT(TEST_TIMEOUT),
#endif // TARGET_OS_WATCH
		T_META_CHECK_LEAKS(false)) {
	dispatch_group_t g = dispatch_group_create();
	for (int i=0; i<30; i++) {
		dispatch_group_async(g, dispatch_get_global_queue(0, 0), ^{
			stress(64*1024, 1000);
		});
	}
	dispatch_group_notify(g, dispatch_get_global_queue(0, 0), ^{
		T_PASS("didn't crash!");
		T_END;
	});
	dispatch_release(g);

	while (1) {
		T_LOG("malloc_zone_pressure_relief");
		malloc_zone_pressure_relief(malloc_default_zone(), 0);
		sleep(1);
	}
}
