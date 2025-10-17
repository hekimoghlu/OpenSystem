/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include <machine/machine_cpu.h>
#include <kern/locks.h>
#include <kern/mpsc_queue.h>
#include <kern/thread.h>

#if !DEBUG && !DEVELOPMENT
#error "Test only file"
#endif

#include <sys/errno.h>

struct mpsc_test_pingpong_queue {
	struct mpsc_daemon_queue queue;
	struct mpsc_queue_chain link;
	struct mpsc_test_pingpong_queue *other;
	uint64_t *count, *end;
};

static void
mpsc_test_pingpong_invoke(mpsc_queue_chain_t elm, __assert_only mpsc_daemon_queue_t dq)
{
	struct mpsc_test_pingpong_queue *q;
	q = mpsc_queue_element(elm, struct mpsc_test_pingpong_queue, link);
	assert(&q->queue == dq);

	if (*q->count % 10000 == 0) {
		printf("mpsc_test_pingpong: %lld asyncs left\n", *q->count);
	}
	if ((*q->count)-- > 0) {
		mpsc_daemon_enqueue(&q->other->queue, &q->other->link,
		    MPSC_QUEUE_DISABLE_PREEMPTION);
	} else {
		*q->end = mach_absolute_time();
		thread_wakeup(&mpsc_test_pingpong_invoke);
	}
}

/*
 * The point of this test is to exercise the enqueue/unlock-drain race
 * since the MPSC queue tries to mimize wakeups when it knows it's useless.
 *
 * It also ensures basic enqueue properties,
 * and will panic if anything goes wrong to help debugging state.
 *
 * Performance wise, we will always go through the wakeup codepath,
 * hence this is mostly a benchmark of
 * assert_wait()/clear_wait()/thread_block()/thread_wakeup()
 * rather than a benchmark of the MPSC queues.
 */
int
mpsc_test_pingpong(uint64_t count, uint64_t *out)
{
	struct mpsc_test_pingpong_queue ping, pong;
	kern_return_t kr;
	wait_result_t wr;
	uint32_t timeout = 5;

	if (count < 1000 || count > 1000 * 1000) {
		return EINVAL;
	}

	printf("mpsc_test_pingpong: START\n");

	kr = mpsc_daemon_queue_init_with_thread(&ping.queue,
	    mpsc_test_pingpong_invoke, MINPRI_KERNEL, "ping",
	    MPSC_DAEMON_INIT_NONE);
	if (kr != KERN_SUCCESS) {
		panic("mpsc_test_pingpong: unable to create pong: %x", kr);
	}

	kr = mpsc_daemon_queue_init_with_thread(&pong.queue,
	    mpsc_test_pingpong_invoke, MINPRI_KERNEL, "pong",
	    MPSC_DAEMON_INIT_NONE);

	if (kr != KERN_SUCCESS) {
		panic("mpsc_test_pingpong: unable to create ping: %x", kr);
	}

	uint64_t n = count, start, end;
	ping.count = pong.count = &n;
	ping.end   = pong.end   = &end;
	ping.other = &pong;
	pong.other = &ping;

#if KASAN
	timeout = 30;
#endif

	assert_wait_timeout(&mpsc_test_pingpong_invoke, THREAD_UNINT,
	    timeout, NSEC_PER_SEC);
	start = mach_absolute_time();
	mpsc_daemon_enqueue(&ping.queue, &ping.link, MPSC_QUEUE_DISABLE_PREEMPTION);

	wr = thread_block(THREAD_CONTINUE_NULL);
	if (wr == THREAD_TIMED_OUT) {
		panic("mpsc_test_pingpong: timed out: ping:%p pong:%p", &ping, &pong);
	}

	printf("mpsc_test_pingpong: CLEANUP\n");

	mpsc_daemon_queue_cancel_and_wait(&ping.queue);
	mpsc_daemon_queue_cancel_and_wait(&pong.queue);
	absolutetime_to_nanoseconds(end - start, out);

	printf("mpsc_test_pingpong: %lld ping-pongs in %lld ns (%lld.%03lld us/async)\n",
	    count, *out, (*out / count) / 1000, (*out / count) % 1000);
	return 0;
}
