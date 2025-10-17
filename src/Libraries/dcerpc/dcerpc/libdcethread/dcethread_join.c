/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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
#include <errno.h>

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

#ifdef API
int interrupt_join(dcethread* thread ATTRIBUTE_UNUSED, void* data);

int
interrupt_join(dcethread* thread ATTRIBUTE_UNUSED, void* data)
{
    dcethread* other = (dcethread*) data;
    if (!pthread_mutex_trylock((pthread_mutex_t*) &other->lock))
    {
        pthread_cond_broadcast((pthread_cond_t*) &other->state_change);
        pthread_mutex_unlock((pthread_mutex_t*) &other->lock);
        return 1;
    }
    else
    {
        return 0;
    }
}

int
dcethread_join(dcethread* thread, void **status)
{
    int (*old_interrupt)(dcethread*, void*);
    void *old_data;

    if (thread == dcethread__self())
    {
        return dcethread__set_errno(EDEADLK);
    }

    if (!thread->flag.joinable)
    {
        DCETHREAD_WARNING("Joining implicit dcethread %p is ill-advised", thread);
    }

    /* Begin our blocking wait on the other thread */
    if (dcethread__begin_block(dcethread__self(), interrupt_join, (void*) thread, &old_interrupt, &old_data))
    {
        dcethread__dispatchinterrupt(dcethread__self());
        return dcethread__set_errno(EINTR);
    }
    /* Lock the other thread in preparation for waiting on its state condition */
    dcethread__lock(thread);
    /* While the thread is still alive */
    while (thread->state != DCETHREAD_STATE_DEAD)
    {
        /* Wait for state change */
        dcethread__wait(thread);
        /* We need to check if we got interrupted, which involves locking dcethread__self().
           To avoid holding two locks simultaneously (which could result in deadlock),
           unlock the other thread for now */
        dcethread__unlock(thread);
        /* Check if we've been interrupted and end block if we have */
        if (dcethread__poll_end_block(dcethread__self(), old_interrupt, old_data))
        {
            /* Process interrupt */
            dcethread__dispatchinterrupt(dcethread__self());
            return dcethread__set_errno(EINTR);
        }
        /* Re-lock thread to resume state change wait */
        dcethread__lock(thread);
    }

    /* Capture thread result */
    if (status)
        *status = thread->status;
    /* Remove reference */
    dcethread__release(thread);
    /* We're done */
    dcethread__unlock(thread);

    return dcethread__set_errno(0);
}

int
dcethread_join_throw(dcethread* thread, void **status)
{
    DCETHREAD_WRAP_THROW(dcethread_join(thread, status));
}

#endif /* API */

#ifdef TEST

#include "dcethread-test.h"

static void*
basic_thread(void* data)
{
    return data;
}

MU_TEST(dcethread_join, basic)
{
    dcethread* thread;
    void* result;

    MU_TRY_DCETHREAD( dcethread_create(&thread, NULL, basic_thread, (void*) 0xDEADBEEF) );
    MU_TRY_DCETHREAD( dcethread_join(thread, &result) );

    MU_ASSERT_EQUAL(MU_TYPE_POINTER, result, (void*) 0xDEADBEEF);
}

static void*
infinite_thread(void* data)
{
    while (1)
    {
        dcethread_pause();
    }

    return NULL;
}

static void*
join_thread(void* data)
{
    dcethread* infinite;
    volatile int interrupted = 0;

    MU_TRY_DCETHREAD( dcethread_create(&infinite, NULL, infinite_thread, NULL) );

    DCETHREAD_TRY
    {
        /* Join will never get anywhere */
        MU_TRY_DCETHREAD( dcethread_join(infinite, NULL) );
    }
    DCETHREAD_CATCH(dcethread_interrupt_e)
    {
        /* Note that we got interrupted */
        interrupted = 1;
        /* Detach and kill thread instead */
        MU_TRY_DCETHREAD( dcethread_interrupt(infinite) );
        MU_TRY_DCETHREAD( dcethread_detach(infinite) );
    }
    DCETHREAD_ENDTRY;

    MU_ASSERT(interrupted);

    return NULL;
}

MU_TEST(dcethread_join, interrupt_pre)
{
    dcethread* thread;

    MU_TRY_DCETHREAD( dcethread_create(&thread, NULL, join_thread, NULL) );
    MU_TRY_DCETHREAD( dcethread_interrupt(thread) );
    MU_TRY_DCETHREAD( dcethread_join(thread, NULL) );
}

MU_TEST(dcethread_join, interrupt_post)
{
    dcethread* thread;
    struct timespec ts;

    ts.tv_nsec = 100000000;
    ts.tv_sec = 0;

    MU_TRY_DCETHREAD( dcethread_create(&thread, NULL, join_thread, NULL) );
    MU_TRY_DCETHREAD( dcethread_delay(&ts) );
    MU_TRY_DCETHREAD( dcethread_interrupt(thread) );
    MU_TRY_DCETHREAD( dcethread_join(thread, NULL) );
}

#endif /* TEST */
