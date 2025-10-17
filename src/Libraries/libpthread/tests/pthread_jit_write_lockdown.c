/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

#include <pthread.h>
#include <sys/wait.h>

static int
test_jit_write_callback(void *ctx)
{
	(void)ctx;
	return 42;
}

// PTHREAD_JIT_WRITE_ALLOW_CALLBACKS_NP deliberately omitted

T_DECL(pthread_jit_write_with_callback_no_allowlist,
		"pthread_jit_write_with_callback_np() without allowlist entitlement",
		T_META_IGNORECRASHES(".*pthread_jit_lockdown.*"))
{
	if (!pthread_jit_write_protect_supported_np()) {
		T_SKIP("JIT write protection not supported on this device");
	}

#if TARGET_OS_OSX
	int rc = pthread_jit_write_with_callback_np(test_jit_write_callback, NULL);
	T_EXPECT_EQ(rc, 42, "pthread_jit_write_with_callback_np");
#else
	pid_t pid = fork();
	if (pid == 0) {
		int rc = pthread_jit_write_with_callback_np(
				test_jit_write_callback, NULL);
		T_ASSERT_FAIL("should not have made it here: %d", rc);
	} else {
		T_ASSERT_POSIX_SUCCESS(pid, "fork()");

		int status;
		pid_t wait_pid = waitpid(pid, &status, 0);
		T_LOG("waitpid(): %d, %d", wait_pid, status);
		T_ASSERT_EQ(wait_pid, pid, "Got child status");
		T_ASSERT_TRUE(WIFSIGNALED(status), "Child terminated by signal");
	}
#endif
}
