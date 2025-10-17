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

#include <pthread.h>
#include <stdio.h>
#include <sys/sysctl.h>

#include "darwintest_defaults.h"

static void *ptr = NULL;

static void destructor(void *value)
{
	ptr = value;
}

static void *thread(void *param)
{
	pthread_key_t key = *(pthread_key_t *)param;
	T_ASSERT_POSIX_ZERO(pthread_setspecific(key, (void *)0x12345678), NULL);
	void *value = pthread_getspecific(key);

	T_ASSERT_POSIX_ZERO(pthread_key_create(&key, NULL), NULL);
	T_ASSERT_POSIX_ZERO(pthread_setspecific(key, (void *)0x55555555), NULL);

	return value;
}

T_DECL(tsd, "tsd",
       T_META_ALL_VALID_ARCHS(YES))
{
	pthread_key_t key;

	T_ASSERT_POSIX_ZERO(pthread_key_create(&key, destructor), NULL);
	T_LOG("key = %ld", key);

	pthread_t p = NULL;
	T_ASSERT_POSIX_ZERO(pthread_create(&p, NULL, thread, &key), NULL);

	void *value = NULL;
	T_ASSERT_POSIX_ZERO(pthread_join(p, &value), NULL);
	T_LOG("value = %p; ptr = %p\n", value, ptr);

	T_EXPECT_EQ(ptr, value, NULL);

	T_ASSERT_POSIX_ZERO(pthread_key_delete(key), NULL);
}

static uint32_t
get_ncpu(void)
{
	static uint32_t activecpu;
	if (!activecpu) {
		uint32_t n;
		size_t s = sizeof(activecpu);
		sysctlbyname("hw.activecpu", &n, &s, NULL, 0);
		activecpu = n;
	}
	return activecpu;
}

T_DECL(cpuid, "cpu id", T_META_ALL_VALID_ARCHS(YES))
{
	pthread_t child;

	size_t cpu_id;
	if (pthread_cpu_number_np(&cpu_id)) {
		T_FAIL("Should not fail to get CPU id");
	}

	T_ASSERT_LE(cpu_id, get_ncpu(), "Got a valid CPU id");
}
