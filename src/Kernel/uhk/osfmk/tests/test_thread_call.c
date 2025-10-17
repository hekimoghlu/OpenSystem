/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
#if !(DEVELOPMENT || DEBUG)
#error "Testing is not enabled on RELEASE configurations"
#endif

#include <tests/xnupost.h>
#include <kern/thread_call.h>
#include <kern/locks.h>
#include <kern/sched_prim.h>

kern_return_t test_thread_call(void);

lck_grp_t test_lock_grp;
lck_mtx_t test_lock;

typedef enum {
	TEST_ARG1 = 0x1234,
	TEST_ARG2 = 0x3456,
} test_param;

int wait_for_callback;
int wait_for_main;

int once_callback_counter = 0;

static void
test_once_callback(thread_call_param_t param0,
    thread_call_param_t param1)
{
	T_ASSERT_EQ_INT((test_param)param0, TEST_ARG1, "param0 is correct");
	T_ASSERT_EQ_INT((test_param)param1, TEST_ARG2, "param1 is correct");

	once_callback_counter++;

	T_ASSERT_EQ_INT(once_callback_counter, 1, "only one callback");

	lck_mtx_lock(&test_lock);

	thread_wakeup(&wait_for_callback);

	uint64_t deadline;
	clock_interval_to_deadline(10, NSEC_PER_SEC, &deadline);

	kern_return_t kr;
	/* wait for the main thread to finish, time out after 10s */
	kr = lck_mtx_sleep_deadline(&test_lock, LCK_SLEEP_DEFAULT, &wait_for_main, THREAD_UNINT, deadline);
	T_ASSERT_EQ_INT(kr, THREAD_AWAKENED, " callback woken by main function");

	lck_mtx_unlock(&test_lock);

	/* sleep for 1s to let the main thread begin the cancel and wait */
	delay_for_interval(1, NSEC_PER_SEC);
}

static void
test_once_thread_call(void)
{
	lck_grp_init(&test_lock_grp, "test_thread_call", LCK_GRP_ATTR_NULL);
	lck_mtx_init(&test_lock, &test_lock_grp, LCK_ATTR_NULL);

	thread_call_t call;
	call = thread_call_allocate_with_options(&test_once_callback,
	    (thread_call_param_t)TEST_ARG1,
	    THREAD_CALL_PRIORITY_HIGH,
	    THREAD_CALL_OPTIONS_ONCE);

	thread_call_param_t arg2_param = (thread_call_param_t)TEST_ARG2;

	lck_mtx_lock(&test_lock);

	thread_call_enter1(call, arg2_param);

	uint64_t deadline;
	clock_interval_to_deadline(10, NSEC_PER_SEC, &deadline);

	kern_return_t kr;
	/* wait for the call to execute, time out after 10s */
	kr = lck_mtx_sleep_deadline(&test_lock, LCK_SLEEP_DEFAULT, &wait_for_callback, THREAD_UNINT, deadline);
	T_ASSERT_EQ_INT(kr, THREAD_AWAKENED, "main function woken by callback");

	lck_mtx_unlock(&test_lock);

	/* at this point the callback is stuck waiting */

	T_ASSERT_EQ_INT(once_callback_counter, 1, "callback fired");

	boolean_t canceled, pending, freed;

	canceled = thread_call_cancel(call);
	T_ASSERT_EQ_INT(canceled, FALSE, "thread_call_cancel should not succeed");

	pending = thread_call_enter1(call, arg2_param);
	T_ASSERT_EQ_INT(pending, FALSE, "call should not be pending");

	/* sleep for 10ms, the call should not execute */
	delay_for_interval(10, NSEC_PER_MSEC);

	canceled = thread_call_cancel(call);
	T_ASSERT_EQ_INT(canceled, TRUE, "thread_call_cancel should succeed");

	pending = thread_call_enter1(call, arg2_param);
	T_ASSERT_EQ_INT(pending, FALSE, "call should not be pending");

	freed = thread_call_free(call);
	T_ASSERT_EQ_INT(freed, FALSE, "thread_call_free should not succeed");

	pending = thread_call_enter1(call, arg2_param);
	T_ASSERT_EQ_INT(pending, TRUE, "call should be pending");

	thread_wakeup(&wait_for_main);

	canceled = thread_call_cancel_wait(call);
	T_ASSERT_EQ_INT(canceled, TRUE, "thread_call_cancel_wait should succeed");

	canceled = thread_call_cancel(call);
	T_ASSERT_EQ_INT(canceled, FALSE, "thread_call_cancel should not succeed");

	freed = thread_call_free(call);
	T_ASSERT_EQ_INT(freed, TRUE, "thread_call_free should succeed");
}

int signal_callback_counter = 0;

static void
test_signal_callback(__unused thread_call_param_t param0,
    __unused thread_call_param_t param1)
{
	/*
	 * ktest sometimes panics if you assert from interrupt context,
	 * and the serial logging will blow past the delay to wait for the interrupt
	 * so don't print in this context.
	 */

	signal_callback_counter++;
}

static void
test_signal_thread_call(void)
{
	thread_call_t call;
	call = thread_call_allocate_with_options(&test_signal_callback,
	    (thread_call_param_t)TEST_ARG1,
	    THREAD_CALL_PRIORITY_HIGH,
	    THREAD_CALL_OPTIONS_ONCE | THREAD_CALL_OPTIONS_SIGNAL);

	thread_call_param_t arg2_param = (thread_call_param_t)TEST_ARG2;

	uint64_t deadline;

	boolean_t canceled, pending, freed;

	clock_interval_to_deadline(10, NSEC_PER_SEC, &deadline);
	pending = thread_call_enter1_delayed(call, arg2_param, deadline);
	T_ASSERT_EQ_INT(pending, FALSE, "call should not be pending");

	canceled = thread_call_cancel(call);
	T_ASSERT_EQ_INT(canceled, TRUE, "thread_call_cancel should succeed");

	clock_interval_to_deadline(10, NSEC_PER_MSEC, &deadline);
	pending = thread_call_enter1_delayed(call, arg2_param, deadline);
	T_ASSERT_EQ_INT(pending, FALSE, "call should not be pending");

	/* sleep for 50ms to let the interrupt fire */
	delay_for_interval(50, NSEC_PER_MSEC);

	T_ASSERT_EQ_INT(signal_callback_counter, 1, "callback fired");

	canceled = thread_call_cancel(call);
	T_ASSERT_EQ_INT(canceled, FALSE, "thread_call_cancel should not succeed");

	freed = thread_call_free(call);
	T_ASSERT_EQ_INT(freed, TRUE, "thread_call_free should succeed");
}

kern_return_t
test_thread_call(void)
{
	test_once_thread_call();
	test_signal_thread_call();

	return KERN_SUCCESS;
}
