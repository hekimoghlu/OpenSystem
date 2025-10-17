/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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
#if DEBUG || DEVELOPMENT

#include <kern/testpoints.h>
#include <kern/exclaves_test_stackshot.h>
#include <kern/thread.h>

// this is set by sysctl
uint64_t tp_scenario;
int32_t  tp_pid = -1;

#define TESTPOINT_COUNT 64
// array of blocked test points
static uint64_t tp_blocked_info[TESTPOINT_COUNT];

static LCK_GRP_DECLARE(tp_lck_grp, "testpoint lock group");
LCK_MTX_DECLARE(tp_mtx, &tp_lck_grp);

void
tp_block(tp_id_t testpoint)
{
	tp_blocked_info[testpoint] = 1;
}

void
tp_unblock(tp_id_t other_testpoint)
{
	tp_blocked_info[other_testpoint] = 0;
	thread_wakeup(&tp_blocked_info);
}


void
tp_wait(tp_id_t testpoint)
{
	wait_result_t wr = THREAD_AWAKENED;
	while ((tp_blocked_info[testpoint]) && wr <= 0) {
		wr = lck_mtx_sleep(&tp_mtx, LCK_SLEEP_DEFAULT, (event_t)&tp_blocked_info, THREAD_INTERRUPTIBLE);
	}
	if (wr > 0) {
		printf("tp_block(%hu) wait interrupted with error %d\n", testpoint, wr);
	}
}

void
tp_relay(tp_id_t testpoint, tp_id_t other_testpoint)
{
	tp_unblock(other_testpoint);
	tp_block(testpoint);
	tp_wait(testpoint);
}


void
tp_call(tp_id_t testpoint, tp_val_t val)
{
	if (tp_pid != pid_from_task(current_task())) {
		return;
	}
	switch (tp_scenario) {
	case TPS_NONE:
		break;
	case TPS_STACKSHOT_UPCALL:
		tp_call_stackshot_upcall(testpoint, val);
		break;
	case TPS_STACKSHOT_LONG_UPCALL:
		tp_call_stackshot_long_upcall(testpoint, val);
		break;
	default:
		panic("Invalid test point scenario value %llu", tp_scenario);
	}
}


static int
testpoint_handler(int64_t testpoint, int64_t *out)
{
	tp_sysctl_msg_t * msg = (tp_sysctl_msg_t*)&testpoint;
	tp_call(msg->id, msg->val);
	*out = 0;
	return 0;
}

SYSCTL_TEST_REGISTER(testpoint, testpoint_handler);

static int
tp_scenario_handler(int64_t scenario, int64_t *out)
{
	tps_id_t new_scenario = (tps_id_t)scenario;

	lck_mtx_lock(&tp_mtx);
	if (tp_scenario != new_scenario) {
		tp_scenario = new_scenario;
		bzero(&tp_blocked_info, sizeof(tp_blocked_info));
		thread_wakeup(&tp_blocked_info);
	}
	lck_mtx_unlock(&tp_mtx);

	printf("tp_scenario=%llu\n", new_scenario);
	*out = 0;
	return 0;
}

SYSCTL_TEST_REGISTER(tp_scenario, tp_scenario_handler);

static int
tp_pid_handler(int64_t pid, int64_t *out)
{
	int32_t new_pid = (int32_t)pid;
	lck_mtx_lock(&tp_mtx);
	if (tp_pid != new_pid) {
		tp_pid = new_pid;
		tp_scenario = TPS_NONE;
		bzero(&tp_blocked_info, sizeof(tp_blocked_info));
		thread_wakeup(&tp_blocked_info);
	}
	lck_mtx_unlock(&tp_mtx);

	printf("tp_pid=%d\n", tp_pid);
	*out = 0;
	return 0;
}

SYSCTL_TEST_REGISTER(tp_pid, tp_pid_handler);

#endif /* DEBUG || DEVELOPMENT */
