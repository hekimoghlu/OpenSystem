/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
/*
 * Timer.c
 * - hides the details of how to get a timer callback
 * - wraps the CFRunLoopTimer functionality
 */

/* 
 * Modification History
 *
 * October 26, 2001	Dieter Siegmund (dieter@apple.com)
 * - created (based on bootp/IPConfiguration.tproj/timer.c)
 */

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <math.h>
#include <CoreFoundation/CFRunLoop.h>
#include "Timer.h"
#include "myCFUtil.h"
#include "mylog.h"


struct Timer_s {
    Timer_func_t *	func;
    void *		arg1;
    void *		arg2;
    void *		arg3;
    void *		rls;
    bool		enabled;
};


#define USECS_PER_SEC		(1000 * 1000)

/**
 ** Timer timer functions
 **/

static void 
Timer_process(CFRunLoopTimerRef rls, void * info)
{
    TimerRef	timer = (TimerRef)info;

    if (timer->func && timer->enabled) {
	timer->enabled = 0;
	(*timer->func)(timer->arg1, timer->arg2, timer->arg3);
    }
    return;
}

TimerRef
Timer_create()
{
    TimerRef timer;

    timer = malloc(sizeof(*timer));
    if (timer == NULL)
	return (NULL);
    bzero(timer, sizeof(*timer));
    return (timer);
}

void
Timer_free(TimerRef * timer_p)
{
    TimerRef timer = *timer_p;

    if (timer == NULL)
	return;

    Timer_cancel(timer);
    free(timer);
    *timer_p = NULL;
    return;
}

int
Timer_set_relative(TimerRef timer, 
		   struct timeval rel_time, Timer_func_t * func, 
		   void * arg1, void * arg2, void * arg3)
{
    CFRunLoopTimerContext 	context =  { 0, NULL, NULL, NULL, NULL };
    CFAbsoluteTime 		wakeup_time;

    if (timer == NULL) {
	return (0);
    }
    Timer_cancel(timer);
    timer->enabled = 0;
    if (func == NULL) {
	return (0);
    }
    timer->func = func;
    timer->arg1 = arg1;
    timer->arg2 = arg2;
    timer->arg3 = arg3;
    timer->enabled = 1;
    if (rel_time.tv_sec <= 0 && rel_time.tv_usec < 1000) {
	rel_time.tv_sec = 0;
	rel_time.tv_usec = 1000; /* force millisecond resolution */
    }
    wakeup_time = CFAbsoluteTimeGetCurrent() + rel_time.tv_sec 
	  + ((double)rel_time.tv_usec / USECS_PER_SEC);
    context.info = timer;
    timer->rls 
	= CFRunLoopTimerCreate(NULL, wakeup_time,
			       0.0, 0, 0,
			       Timer_process,
			       &context);
    CFRunLoopAddTimer(CFRunLoopGetCurrent(), timer->rls,
		      kCFRunLoopDefaultMode);
    return (1);
}

void
Timer_cancel(TimerRef timer)
{
    if (timer == NULL) {
	return;
    }
    timer->enabled = 0;
    timer->func = NULL;
    if (timer->rls) {
	CFRunLoopTimerInvalidate(timer->rls);
	my_CFRelease(&timer->rls);
    }
    return;
}

/**
 ** timer functions
 **/
long
Timer_current_secs()
{
    return ((long)CFAbsoluteTimeGetCurrent());
}

struct timeval
Timer_current_time()
{
    double 		t = CFAbsoluteTimeGetCurrent();
    struct timeval 	tv;

    tv.tv_sec = (int32_t)t;
    tv.tv_usec = (t - tv.tv_sec) * USECS_PER_SEC;

    return (tv);
}
