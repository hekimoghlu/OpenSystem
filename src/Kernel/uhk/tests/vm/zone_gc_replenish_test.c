/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#include <sys/sysctl.h>
#include <time.h>

#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("VM"));

static int64_t
run_sysctl_test(const char *t, int64_t value)
{
	char name[1024];
	int64_t result = 0;
	size_t s = sizeof(value);
	int rc;

	snprintf(name, sizeof(name), "debug.test.%s", t);
	rc = sysctlbyname(name, &result, &s, &value, s);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(rc, "sysctlbyname(%s)", t);
	return result;
}


static void *
gc_thread_func(__unused void *arg)
{
	time_t start = time(NULL);
	size_t n = 0;

	/*
	 * Keep kicking the test for 15 seconds to see if we can panic() the kernel
	 */
	while (time(NULL) < start + 15) {
		run_sysctl_test("zone_gc_replenish_test", 0);
		if (++n % 100000 == 0) {
			T_LOG("%zd zone_gc_replenish_test done", n);
		}
	}
	return NULL;
}

static void *
alloc_thread_func(__unused void *arg)
{
	time_t start = time(NULL);
	size_t n = 0;

	/*
	 * Keep kicking the test for 15 seconds to see if we can panic() the kernel
	 */
	while (time(NULL) < start + 15) {
		run_sysctl_test("zone_alloc_replenish_test", 0);
		if (++n % 10000 == 0) {
			T_LOG("%zd zone_alloc_replenish_test done", n);
		}
	}
	return NULL;
}

T_DECL(zone_gc_replenish_test,
    "Test zone garbage collection, exhaustion and replenishment",
    T_META_CHECK_LEAKS(false), T_META_TAG_VM_PREFERRED)
{
	pthread_attr_t attr;
	pthread_t gc_thread;
	pthread_t alloc_thread;
	int ret;

	ret = pthread_attr_init(&attr);
	T_QUIET; T_ASSERT_MACH_SUCCESS(ret, "pthread_attr_init");

	ret = pthread_create(&gc_thread, &attr, gc_thread_func, NULL);
	T_QUIET; T_ASSERT_POSIX_ZERO(ret, "gc pthread_create");

	ret = pthread_create(&alloc_thread, &attr, alloc_thread_func, NULL);
	T_QUIET; T_ASSERT_POSIX_ZERO(ret, "alloc pthread_create");

	T_ASSERT_POSIX_ZERO(pthread_join(gc_thread, NULL), NULL);
	T_ASSERT_POSIX_ZERO(pthread_join(alloc_thread, NULL), NULL);
	T_PASS("Ran 15 seconds with no panic");
}
