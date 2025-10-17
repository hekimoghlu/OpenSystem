/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
 * devtimer.h
 * - timer source based on <kern/thread_call.h>
 */


#ifndef _NET_DEVTIMER_H
#define _NET_DEVTIMER_H

#include <sys/types.h>
#include <sys/systm.h>

#define DEVTIMER_USECS_PER_SEC          (1000 * 1000)

enum {
	devtimer_process_func_event_lock,
	devtimer_process_func_event_unlock,
};
typedef int devtimer_process_func_event;

typedef struct devtimer_s * devtimer_ref;
typedef void (*devtimer_process_func)(devtimer_ref timer,
    devtimer_process_func_event event);
typedef void (*devtimer_timeout_func)(void * arg0, void * arg1, void * arg2);

int
devtimer_valid(devtimer_ref timer);

void
devtimer_retain(devtimer_ref timer);

void *
devtimer_arg0(devtimer_ref timer);

devtimer_ref
devtimer_create(devtimer_process_func process_func, void * arg0);

void
devtimer_invalidate(devtimer_ref timer);

void
devtimer_release(devtimer_ref timer);

void
devtimer_set_absolute(devtimer_ref t,
    struct timeval abs_time,
    devtimer_timeout_func func,
    void * arg1, void * arg2);

void
devtimer_set_relative(devtimer_ref t,
    struct timeval rel_time,
    devtimer_timeout_func func,
    void * arg1, void * arg2);
void
devtimer_cancel(devtimer_ref t);

int
devtimer_enabled(devtimer_ref t);

struct timeval
devtimer_current_time(void);

int32_t
devtimer_current_secs(void);

#endif /* _NET_DEVTIMER_H */
