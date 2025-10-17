/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#include <kperf/task_samplers.h>
#include <kperf/context.h>
#include <kperf/buffer.h>
#include <kern/thread.h>

#include <kern/task.h>

extern void memorystatus_proc_flags_unsafe(void * v, boolean_t *is_dirty,
    boolean_t *is_dirty_tracked, boolean_t *allow_idle_exit);

void
kperf_task_snapshot_sample(task_t task, struct kperf_task_snapshot *tksn)
{
	BUF_INFO(PERF_TK_SNAP_SAMPLE | DBG_FUNC_START);

	assert(tksn != NULL);

	tksn->kptksn_flags = 0;
	if (task->effective_policy.tep_darwinbg) {
		tksn->kptksn_flags |= KPERF_TASK_FLAG_DARWIN_BG;
	}
	if (task->requested_policy.trp_role == TASK_FOREGROUND_APPLICATION) {
		tksn->kptksn_flags |= KPERF_TASK_FLAG_FOREGROUND;
	}
	if (task->requested_policy.trp_boosted == 1) {
		tksn->kptksn_flags |= KPERF_TASK_FLAG_BOOSTED;
	}
#if CONFIG_MEMORYSTATUS
	boolean_t dirty = FALSE, dirty_tracked = FALSE, allow_idle_exit = FALSE;
	memorystatus_proc_flags_unsafe(get_bsdtask_info(task), &dirty, &dirty_tracked, &allow_idle_exit);
	if (dirty) {
		tksn->kptksn_flags |= KPERF_TASK_FLAG_DIRTY;
	}
	if (dirty_tracked) {
		tksn->kptksn_flags |= KPERF_TASK_FLAG_DIRTY_TRACKED;
	}
	if (allow_idle_exit) {
		tksn->kptksn_flags |= KPERF_TASK_ALLOW_IDLE_EXIT;
	}
#endif

	tksn->kptksn_suspend_count = task->suspend_count;
	tksn->kptksn_pageins = (integer_t) MIN(counter_load(&task->pageins), INT32_MAX);
	struct recount_times_mach times = recount_task_terminated_times(task);
	tksn->kptksn_user_time_in_terminated_threads = times.rtm_user;
	tksn->kptksn_system_time_in_terminated_threads = times.rtm_system;

	BUF_INFO(PERF_TK_SNAP_SAMPLE | DBG_FUNC_END);
}

void
kperf_task_snapshot_log(struct kperf_task_snapshot *tksn)
{
	assert(tksn != NULL);

#if defined(__LP64__)
	BUF_DATA(PERF_TK_SNAP_DATA, tksn->kptksn_flags,
	    ENCODE_UPPER_64(tksn->kptksn_suspend_count) |
	    ENCODE_LOWER_64(tksn->kptksn_pageins),
	    tksn->kptksn_user_time_in_terminated_threads,
	    tksn->kptksn_system_time_in_terminated_threads);
#else
	BUF_DATA(PERF_TK_SNAP_DATA1_32, UPPER_32(tksn->kptksn_flags),
	    LOWER_32(tksn->kptksn_flags),
	    tksn->kptksn_suspend_count,
	    tksn->kptksn_pageins);
	BUF_DATA(PERF_TK_SNAP_DATA2_32, UPPER_32(tksn->kptksn_user_time_in_terminated_threads),
	    LOWER_32(tksn->kptksn_user_time_in_terminated_threads),
	    UPPER_32(tksn->kptksn_system_time_in_terminated_threads),
	    LOWER_32(tksn->kptksn_system_time_in_terminated_threads));
#endif /* defined(__LP64__) */
}

void
kperf_task_info_log(struct kperf_context *ctx)
{
	assert(ctx != NULL);

	BUF_DATA(PERF_TK_INFO_DATA, ctx->cur_pid);
}
