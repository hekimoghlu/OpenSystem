/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#include <stdint.h>

#include <kern/thread.h>

#include <kperf/action.h>
#include <kperf/buffer.h>
#include <kperf/kperf.h>
#include <kperf/lazy.h>
#include <kperf/sample.h>

unsigned int kperf_lazy_wait_action = 0;
unsigned int kperf_lazy_cpu_action = 0;
uint64_t kperf_lazy_wait_time_threshold = 0;
uint64_t kperf_lazy_cpu_time_threshold = 0;

void
kperf_lazy_reset(void)
{
	kperf_lazy_wait_action = 0;
	kperf_lazy_wait_time_threshold = 0;
	kperf_lazy_cpu_action = 0;
	kperf_lazy_cpu_time_threshold = 0;
	kperf_on_cpu_update();
}

void
kperf_lazy_off_cpu(thread_t thread)
{
	/* try to lazily sample the CPU if the thread was pre-empted */
	if ((thread->reason & AST_SCHEDULING) != 0) {
		kperf_lazy_cpu_sample(thread, 0, 0);
	}
}

void
kperf_lazy_make_runnable(thread_t thread, bool in_interrupt)
{
	assert(thread->last_made_runnable_time != THREAD_NOT_RUNNABLE);
	/* ignore threads that race to wait and in waking up */
	if (thread->last_run_time > thread->last_made_runnable_time) {
		return;
	}

	uint64_t wait_time = thread_get_last_wait_duration(thread);
	if (wait_time > kperf_lazy_wait_time_threshold) {
		BUF_DATA(PERF_LZ_MKRUNNABLE, (uintptr_t)thread_tid(thread),
		    thread->sched_pri, in_interrupt ? 1 : 0);
	}
}

void
kperf_lazy_wait_sample(thread_t thread, thread_continue_t continuation,
    uintptr_t *starting_fp)
{
	/* ignore idle threads */
	if (thread->last_made_runnable_time == THREAD_NOT_RUNNABLE) {
		return;
	}
	/* ignore invalid made runnable times */
	if (thread->last_made_runnable_time < thread->last_run_time) {
		return;
	}

	/* take a sample if thread was waiting for longer than threshold */
	uint64_t wait_time = thread_get_last_wait_duration(thread);
	if (wait_time > kperf_lazy_wait_time_threshold) {
		uint64_t runnable_time = timer_grab(&thread->runnable_timer);
		uint64_t running_time = recount_thread_time_mach(thread);

		BUF_DATA(PERF_LZ_WAITSAMPLE, wait_time, runnable_time, running_time);

		task_t task = get_threadtask(thread);
		struct kperf_context ctx = {
			.cur_thread = thread,
			.cur_task = task,
			.cur_pid = task_pid(task),
			.trigger_type = TRIGGER_TYPE_LAZY_WAIT,
			.starting_fp = starting_fp,
		};

		struct kperf_sample *sample = kperf_intr_sample_buffer();
		if (!sample) {
			return;
		}

		unsigned int flags = SAMPLE_FLAG_PEND_USER;
		flags |= continuation ? SAMPLE_FLAG_CONTINUATION : 0;
		flags |= !ml_at_interrupt_context() ? SAMPLE_FLAG_NON_INTERRUPT : 0;

		kperf_sample(sample, &ctx, kperf_lazy_wait_action, flags);
	}
}

void
kperf_lazy_cpu_sample(thread_t thread, unsigned int flags, bool interrupt)
{
	assert(ml_get_interrupts_enabled() == FALSE);
	if (!thread) {
		thread = current_thread();
	}

	/* take a sample if this CPU's last sample time is beyond the threshold */
	processor_t processor = current_processor();
#if __arm__ || __arm64__
	uint64_t time_now = ml_get_speculative_timebase();
#else // __arm__ || __arm64__
	uint64_t time_now = mach_absolute_time();
#endif // !__arm__ && !__arm64__

	uint64_t since_last_sample = time_now - processor->kperf_last_sample_time;
	if (since_last_sample > kperf_lazy_cpu_time_threshold) {
		processor->kperf_last_sample_time = time_now;
		uint64_t runnable_time = timer_grab(&thread->runnable_timer);
		uint64_t running_time = recount_thread_time_mach(thread);

		BUF_DATA(PERF_LZ_CPUSAMPLE, running_time, runnable_time,
		    thread->sched_pri, interrupt ? 1 : 0);

		task_t task = get_threadtask(thread);
		struct kperf_context ctx = {
			.cur_thread = thread,
			.cur_task = task,
			.cur_pid = task_pid(task),
			.trigger_type = TRIGGER_TYPE_LAZY_CPU,
			.starting_fp = 0,
		};

		struct kperf_sample *sample = kperf_intr_sample_buffer();
		if (!sample) {
			return;
		}

		kperf_sample(sample, &ctx, kperf_lazy_cpu_action,
		    SAMPLE_FLAG_PEND_USER | flags);
	}
}

/*
 * Accessors for configuration.
 */

int
kperf_lazy_get_wait_action(void)
{
	return kperf_lazy_wait_action;
}

int
kperf_lazy_set_wait_action(int action_id)
{
	if (action_id < 0 || (unsigned int)action_id > kperf_action_get_count()) {
		return 1;
	}

	kperf_lazy_wait_action = action_id;
	kperf_on_cpu_update();
	return 0;
}

uint64_t
kperf_lazy_get_wait_time_threshold(void)
{
	return kperf_lazy_wait_time_threshold;
}

int
kperf_lazy_set_wait_time_threshold(uint64_t threshold)
{
	kperf_lazy_wait_time_threshold = threshold;
	return 0;
}

int
kperf_lazy_get_cpu_action(void)
{
	return kperf_lazy_cpu_action;
}

int
kperf_lazy_set_cpu_action(int action_id)
{
	if (action_id < 0 || (unsigned int)action_id > kperf_action_get_count()) {
		return 1;
	}

	kperf_lazy_cpu_action = action_id;
	return 0;
}

uint64_t
kperf_lazy_get_cpu_time_threshold(void)
{
	return kperf_lazy_cpu_time_threshold;
}

int
kperf_lazy_set_cpu_time_threshold(uint64_t threshold)
{
	kperf_lazy_cpu_time_threshold = threshold;
	return 0;
}
