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
#include <darwintest_multiprocess.h>
#include <notify.h>
#include <notify_private.h>
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <unistd.h>
#include <sandbox.h>
#include <TargetConditionals.h>
#include <stdio.h>
#include <signal.h>

#define KEY "com.apple.notify.test-loopback"

#define LOG(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#define FAIL_LOG(fmt, ...) LOG("[FAIL] " fmt, ##__VA_ARGS__)
#define PASS_LOG(fmt, ...) LOG("[PASS] " fmt, ##__VA_ARGS__)

#define ASSERT_EQ(a, b, fmt, ...) ASSERT((a) == (b), fmt, ##__VA_ARGS__)
#define ASSERT(expr, fmt, ...) \
if (!(expr)) { \
	FAIL_LOG(fmt, ##__VA_ARGS__);\
	if (test_result == 0) test_result = __LINE__; \
} else {\
	PASS_LOG(fmt, ##__VA_ARGS__);\
}
#define ASSERT_EQ(a, b, fmt, ...) ASSERT((a) == (b), fmt, ##__VA_ARGS__)

static int
run_test()
{
	uint32_t rc;
	int c_token, d_token, check;
	int ret, sig;
	uint64_t state = 0;
	dispatch_semaphore_t sema = dispatch_semaphore_create(0);
	dispatch_queue_t dq = dispatch_queue_create_with_target("q", NULL, NULL);
	__block int test_result = 0;

	// Simulate WebContent notify options
	notify_set_options(NOTIFY_OPT_DISPATCH | NOTIFY_OPT_REGEN | NOTIFY_OPT_FILTERED);

#if !TARGET_OS_OSX
	printf("Skipping entering sandbox.\n");
#else // !TARGET_OS_OSX
	char *sberr = NULL;
	const char *sandbox_name = "com.apple.notify-test-lockdown";
	ret = sandbox_init(sandbox_name, SANDBOX_NAMED, &sberr);
	ASSERT(ret >= 0, "sandbox_init %s: %s", sandbox_name, sberr);
	sandbox_free_error(sberr);
#endif

	rc = notify_register_check(KEY, &c_token);
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_register_check successful");

	rc = notify_check(c_token, &check);
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_check successful");
	ASSERT_EQ(check, 1, "notify_check successful");

	rc = notify_check(c_token, &check);
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_check successful");
	ASSERT_EQ(check, 0, "notify_check successful");

	rc = notify_get_state(c_token, &state);
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_get_state successful");
	ASSERT(state != 0xAAAAAAAA, "matches expected state 0x%llx", state);

	rc = notify_register_dispatch(KEY, &d_token, dq, ^(int token){
		uint64_t state;
		uint32_t status;

		status = notify_get_state(token, &state);
		ASSERT_EQ(status, NOTIFY_STATUS_OK, "notify_get_state successful");
		ASSERT(state == 0xDEADBEEF, "matches expected state 0x%llx", state);
		dispatch_semaphore_signal(sema);
	});
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_register_dispatch successful");

	sleep(1);

	rc = notify_set_state(d_token, 0xDEADBEEF);
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_set_state successful");

	rc = notify_post(KEY);
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_post successful");

	dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);

	sigset_t sigset;
	ret = sigemptyset(&sigset);
	ASSERT_EQ(ret, 0, "sigemptyset successful");

	ret = sigaddset(&sigset, SIGUSR1);
	ASSERT_EQ(ret, 0, "sigaddset successful");

	ret = sigwait(&sigset, &sig);
	ASSERT_EQ(ret, 0, "sigwait successful");
	ASSERT_EQ(sig, SIGUSR1, "matches expected signal");

	rc = notify_cancel(c_token);
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_cancel successful");

	rc = notify_cancel(d_token);
	ASSERT_EQ(rc, NOTIFY_STATUS_OK, "notify_cancel successful");

	return test_result;
}

int
main(int argc, const char *argv[])
{
	return run_test();
}
