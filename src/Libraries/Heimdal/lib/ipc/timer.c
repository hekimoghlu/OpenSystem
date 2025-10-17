/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#include "hi_locl.h"
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>

#include "heap.h"

struct heim_event_data {
    heap_ptr hptr;
    dispatch_semaphore_t running;
    int flags;
#define RUNNING	1
#define IN_FREE 2
#define CF_OBJECT 4
    heim_ipc_event_callback_t callback;
    heim_ipc_event_final_t final;
    void *ctx;
    time_t t;
};

/**
 * Event handling framework
 *
 * Event lifesyncle
 *
 * create ---> set_time ------> do_event --------> delete_event
 *     |         |                           |
 *      \--------\-------> cancel_event -->--/
 *
 */

static dispatch_queue_t timer_sync_q;
static dispatch_queue_t timer_job_q;
static Heap *timer_heap;
static dispatch_source_t timer_source;

/*
 * Compare to event for heap sorting
 */

static int
event_cmp_fn(const void *aptr, const void *bptr)
{
    const struct heim_event_data *a = aptr;
    const struct heim_event_data *b = bptr;
    return (int)(a->t - b->t);
}

/*
 * Calculate next timer event and set the timer
 */

static void
reschedule_timer(void)
{
    const struct heim_event_data *e = heap_head(timer_heap);

    if (e == NULL) {
	/*
	 * if there are no more events, cancel timer by setting timer
	 * to forever, later calls will pull it down to !forever when
	 * needed again
	 */
	dispatch_source_set_timer(timer_source,
				  DISPATCH_TIME_FOREVER, DISPATCH_TIME_FOREVER, 0);
    } else {
	struct timespec ts;
	ts.tv_sec = e->t;
	ts.tv_nsec = 0;
	dispatch_source_set_timer(timer_source,
				  dispatch_walltime(&ts, 0),
				  DISPATCH_TIME_FOREVER,
				  10ull * NSEC_PER_SEC);
    }
}

/*
 * Get jobs that have triggered and run them in the background.
 */

static void
trigger_jobs(void)
{
    time_t now = time(NULL);

    while (1) {
	struct heim_event_data *e = rk_UNCONST(heap_head(timer_heap));

	if (e != NULL && e->t < now) {
	    heap_remove_head(timer_heap);
	    e->hptr = HEAP_INVALID_PTR;

	    /* if its already running, lets retry 10s from now */
	    if (e->flags & RUNNING) {
		e->t = now + 10;
		heap_insert(timer_heap, e, &e->hptr);
		continue;
	    }
	    e->flags |= RUNNING;

	    _heim_ipc_suspend_timer();

	    dispatch_async(timer_job_q, ^{
		    e->callback(e, e->ctx);
		    dispatch_async(timer_sync_q, ^{
			    e->flags &= ~RUNNING;
			    if (e->running)
				dispatch_semaphore_signal(e->running);

			    _heim_ipc_restart_timer();
			});
		});
	} else
	    break;
    }
    reschedule_timer();
}

/*
 * Create sync syncronization queue, heap and timer
 */

static void
timer_init(void)
{
    static dispatch_once_t once;

    dispatch_once(&once, ^{ 

	    timer_sync_q = dispatch_queue_create("hiem-timer-q", NULL);
	    timer_job_q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);


	    timer_heap = heap_new(11, event_cmp_fn);

	    timer_source = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER,
						  0, 0, timer_sync_q);
	    dispatch_source_set_event_handler(timer_source, ^{ trigger_jobs(); });
	    dispatch_resume(timer_source);
	});
}

/**
 * Create a event that is (re)schedule and set the callback functions
 * and context variable.
 *
 * The callback function can call heim_ipc_event_cancel() and
 * heim_ipc_event_free().
 *
 * @param cb callback function when the event is triggered
 * @param ctx context passed to the callback function
 *
 * @return a heim ipc event
 */

heim_event_t
heim_ipc_event_create_f(heim_ipc_event_callback_t cb, void *ctx)
{
    heim_event_t e;

    timer_init();

    e = malloc(sizeof(*e));
    if (e == NULL)
	return NULL;

    e->hptr = HEAP_INVALID_PTR;
    e->running = NULL;
    e->flags = 0;
    e->callback = cb;
    e->ctx = ctx;
    e->t = 0;

    return e;
}

/**
 * Create a event that is (re)schedule and set the callback functions
 * and context variable.
 *
 * The callback function can call heim_ipc_event_cancel() and
 * heim_ipc_event_free().
 *
 * @param cb callback function when the event is triggered
 * @param ctx CFTypeRef context passed to the callback function, retained until freed.
 *
 * @return a heim ipc event
 */

heim_event_t
heim_ipc_event_cf_create_f(heim_ipc_event_callback_t cb, CFTypeRef ctx)
{
    CFRetain(ctx);
    heim_event_t e = heim_ipc_event_create_f(cb, (void*)ctx);
    e->flags |= CF_OBJECT;
    
    return e;
}

/**
 * (Re)schedule a new timeout for an event
 *
 * @param e event to schedule new timeout
 * @param t absolute time the event will trigger
 *
 * @return 0 on success
 */

int
heim_ipc_event_set_time(heim_event_t e, time_t t)
{
    dispatch_sync(timer_sync_q, ^{
	    time_t next;
	    if (e->flags & IN_FREE)
		abort();
	    if (e->hptr != HEAP_INVALID_PTR)
		heap_remove(timer_heap, e->hptr);

	    next = time(NULL);

	    /* don't allow setting events in the past */
	    if (t > next)
		next = t;
	    e->t = next;

	    heap_insert(timer_heap, e, &e->hptr);
	    reschedule_timer();
	});
    return 0;
}

/**
 * Cancel an event.
 *
 * Cancel will block if the callback for the job is running.
 *
 * @param e event to schedule new timeout
 */

void
heim_ipc_event_cancel(heim_event_t e)
{
    dispatch_sync(timer_sync_q, ^{
	    if (e->hptr != HEAP_INVALID_PTR) {
		heap_remove(timer_heap, e->hptr);
		e->hptr = HEAP_INVALID_PTR;
	    }
	    e->t = 0;
	    reschedule_timer();
	});
}

bool
heim_ipc_event_is_cancelled(heim_event_t e)
{
    __block bool result = false;
    dispatch_sync(timer_sync_q, ^{
	result = e->t == 0;
    });
    return result;
}

/**
 * Free an event, most be either canceled or triggered. Can't delete
 * an event that is not canceled.
 *
 * @param e event to free
 */

void
heim_ipc_event_free(heim_event_t e)
{
    dispatch_async(timer_sync_q, ^{
	    e->flags |= IN_FREE;
	    if ((e->hptr != HEAP_INVALID_PTR))
		abort();
	    if (e->final || (e->flags & RUNNING)) {
		int wait_running = (e->flags & RUNNING);

		if (wait_running)
		    e->running = dispatch_semaphore_create(0);

		dispatch_async(timer_job_q, ^{
			if (wait_running) {
			    dispatch_semaphore_wait(e->running,
						    DISPATCH_TIME_FOREVER);
			    dispatch_release(e->running);
			}
			if (e->final)
			    e->final(e->ctx);
			if (e->flags & CF_OBJECT) {
			    CFRelease(e->ctx);
			}
			free(e);
		    });
	    } else {
		if (e->flags & CF_OBJECT) {
		    CFRelease(e->ctx);
		}
		free(e);
	    }
	});
}

/**
 * Finalizer called when event 'e' is freed.
 *
 * @param e event to set finalizer for
 * @param f finalizer to be called
 */

void
heim_ipc_event_set_final_f(heim_event_t e, heim_ipc_event_final_t f)
{
    e->final = f;
}
