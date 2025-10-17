/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#include <darwintest_utils.h>

#include "pthread_jit_write_test_inline.h"

int
test_dylib_jit_write_callback(void *ctx);

static int
test_jit_write_callback_1(void *ctx)
{
	T_EXPECT_EQ(does_access_fault(ACCESS_WRITE, rwx_addr), false,
			"Do not expect write access to fault in write callback");

	return *(int *)ctx;
}

static int
test_jit_write_callback_2(void *ctx)
{
	T_EXPECT_EQ(does_access_fault(ACCESS_WRITE, rwx_addr), false,
			"Do not expect write access to fault in write callback 2");

	return 0;
}

static int
dylib_callback_callback(void *ctx)
{
	T_EXPECT_EQ(does_access_fault(ACCESS_WRITE, rwx_addr), false,
			"Write access should also not fault in dylib callback");
	return 4242;
}

PTHREAD_JIT_WRITE_ALLOW_CALLBACKS_NP(test_jit_write_callback_1,
		test_jit_write_callback_2);

T_DECL(pthread_jit_write_with_callback_allowed,
		"Verify pthread_jit_write_with_callback_np(3) works correctly when used correctly")
{
	bool expect_fault = pthread_jit_write_protect_supported_np();

	pthread_jit_test_setup();

	T_EXPECT_EQ(does_access_fault(ACCESS_WRITE, rwx_addr), expect_fault,
			"If supported, write should initially fault");

	int write_ctx = 42;
	int rc = pthread_jit_write_with_callback_np(test_jit_write_callback_1,
			&write_ctx);
	T_EXPECT_EQ(rc, 42, "Callback had expected return value");

	T_EXPECT_EQ(does_access_fault(ACCESS_EXECUTE, rwx_addr), false,
			"Do not expect execute access to fault after write callback");

	T_EXPECT_EQ(does_access_fault(ACCESS_WRITE, rwx_addr), expect_fault,
			"Write access should fault outside of callback if supported");

	// test that more than one callback can be allowed
	rc = pthread_jit_write_with_callback_np(test_jit_write_callback_2,
			NULL);
	T_EXPECT_EQ(rc, 0, "Callback had expected return value");

	// test that callbacks in dylibs can be allowed
	rc = pthread_jit_write_with_callback_np(test_dylib_jit_write_callback,
			dylib_callback_callback);
	T_EXPECT_EQ(rc, 4242, "Callback callback returned expected result");

	pthread_jit_test_teardown();
}

#if TARGET_CPU_ARM64 // should effectively match _PTHREAD_CONFIG_JIT_WRITE_PROTECT

#define HELPER_TOOL_PATH "/AppleInternal/Tests/libpthread/assets/pthread_jit_write_with_callback_tool"

#if TARGET_OS_OSX
T_DECL(pthread_jit_write_protect_np_disallowed,
		"Verify pthread_jit_write_protect_np prohibited by allowlist",
		T_META_IGNORECRASHES(".*pthread_jit_write_with_callback_tool.*"))
{
	if (!pthread_jit_write_protect_supported_np()) {
		T_SKIP("JIT write protection not supported on this device");
	}

	char *cmd[] = { HELPER_TOOL_PATH, "pthread_jit_write_protect_np", NULL };
	dt_spawn_t spawn = dt_spawn_create(NULL);
	dt_spawn(spawn, cmd, 
			^(char *line, __unused size_t size){
				T_LOG("+ %s", line);
			},
			^(char *line, __unused size_t size){
				T_LOG("stderr: %s\n", line);
			});

	bool exited, signaled;
	int status, signal;
	dt_spawn_wait(spawn, &exited, &signaled, &status, &signal);
	T_EXPECT_FALSE(exited, "helper tool should not have exited");
	T_EXPECT_TRUE(signaled, "helper tool should have been signaled");
}
#endif // TARGET_OS_OSX

T_DECL(pthread_jit_write_with_invalid_callback_disallowed,
		"Verify pthread_jit_write_with_callback fails bad callbacks",
		T_META_IGNORECRASHES(".*pthread_jit_write_with_callback_tool.*"))
{
	if (!pthread_jit_write_protect_supported_np()) {
		T_SKIP("JIT write protection not supported on this device");
	}

	char *cmd[] = { HELPER_TOOL_PATH, "pthread_jit_write_with_callback_np", NULL };
	dt_spawn_t spawn = dt_spawn_create(NULL);
	dt_spawn(spawn, cmd, 
			^(char *line, __unused size_t size){
				T_LOG("+ %s", line);
			},
			^(char *line, __unused size_t size){
				T_LOG("stderr: %s\n", line);
			});

	bool exited, signaled;
	int status, signal;
	dt_spawn_wait(spawn, &exited, &signaled, &status, &signal);
	T_EXPECT_FALSE(exited, "helper tool should not have exited");
	T_EXPECT_TRUE(signaled, "helper tool should have been signaled");
}

#endif // TARGET_CPU_ARM64
