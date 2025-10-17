/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include <config.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

int
dcethread_cond_timedwait(dcethread_cond *cond, dcethread_mutex *mutex, struct timespec *abstime)
{
    int ret = 0;
    int (*interrupt_old)(dcethread*, void*) = NULL;
    void *data_old = NULL;
    condwait_info info;

    info.cond = cond;
    info.mutex = mutex;

    do
    {
        if (dcethread__begin_block(dcethread__self(), dcethread__interrupt_condwait, &info, &interrupt_old, &data_old))
        {
            dcethread__dispatchinterrupt(dcethread__self());
            return dcethread__set_errno(EINTR);
        }
        mutex->owner = (pthread_t) -1;
	ret = pthread_cond_timedwait(cond, (pthread_mutex_t*) &mutex->mutex, abstime);
        mutex->owner = pthread_self();
        if (dcethread__end_block(dcethread__self(), interrupt_old, data_old))
        {
            dcethread__dispatchinterrupt(dcethread__self());
            return dcethread__set_errno(EINTR);
        }
    } while (ret == EINTR);

    return dcethread__set_errno(ret);
}

int
dcethread_cond_timedwait_throw(dcethread_cond *cond, dcethread_mutex *mutex, struct timespec *abstime)
{
    int ret = dcethread_cond_timedwait(cond, mutex, abstime);

    if (ret < 0 && errno == ETIMEDOUT)
	return -1;
    else if (ret < 0)
    {
        dcethread__exc_raise(dcethread__exc_from_errno(errno), __FILE__, __LINE__);
    }
    else
    {
        return ret;
    }
}
