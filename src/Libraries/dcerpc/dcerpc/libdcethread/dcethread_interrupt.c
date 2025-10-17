/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

int
dcethread_interrupt(dcethread* thread)
{
    dcethread__lock(thread);
    dcethread__interrupt(thread);
    dcethread__unlock(thread);

    return dcethread__set_errno(0);
}

int
dcethread_interrupt_throw(dcethread* thread)
{
    DCETHREAD_WRAP_THROW(dcethread_interrupt(thread));
}

#endif /* API */

#ifdef TEST

#include "dcethread-test.h"

static void*
basic_thread(void* data)
{
    volatile int interrupt_caught = 0;

    DCETHREAD_TRY
    {
	MU_ASSERT(!interrupt_caught);
	while (1)
	{
	    dcethread_checkinterrupt();
	    dcethread_yield();
	}
    }
    DCETHREAD_CATCH(dcethread_interrupt_e)
    {
	MU_ASSERT(!interrupt_caught);
	interrupt_caught = 1;
    }
    DCETHREAD_ENDTRY;

    MU_ASSERT(interrupt_caught);

    return NULL;
}

MU_TEST(dcethread_interrupt, basic)
{
    dcethread* thread;

    MU_TRY_DCETHREAD( dcethread_create(&thread, NULL, basic_thread, NULL) );
    MU_TRY_DCETHREAD( dcethread_interrupt(thread) );
    MU_TRY_DCETHREAD( dcethread_join(thread, NULL) );
}

MU_TEST(dcethread_interrupt, self)
{
    volatile int interrupt_caught = 0;

    DCETHREAD_TRY
    {
        MU_ASSERT(!interrupt_caught);
        dcethread_interrupt(dcethread_self());
        dcethread_checkinterrupt();
    }
    DCETHREAD_CATCH(dcethread_interrupt_e)
    {
        MU_ASSERT(!interrupt_caught);
        interrupt_caught = 1;
    }
    DCETHREAD_ENDTRY;

    MU_ASSERT(interrupt_caught);
}

MU_TEST(dcethread_interrupt, disable)
{
    DCETHREAD_TRY
    {
        dcethread_enableinterrupt(0);
        dcethread_interrupt(dcethread_self());
        dcethread_checkinterrupt();
    }
    DCETHREAD_CATCH(dcethread_interrupt_e)
    {
        MU_ASSERT_NOT_REACHED();
    }
    DCETHREAD_ENDTRY;
}

MU_TEST(dcethread_interrupt, disable_enable)
{
    volatile int interrupt_caught = 0;

    DCETHREAD_TRY
    {
        dcethread_enableinterrupt(0);
        dcethread_interrupt(dcethread_self());
        dcethread_enableinterrupt(1);
        dcethread_checkinterrupt();
    }
    DCETHREAD_CATCH(dcethread_interrupt_e)
    {
        MU_ASSERT(!interrupt_caught);
        interrupt_caught = 1;
    }
    DCETHREAD_ENDTRY;

    MU_ASSERT(interrupt_caught);
}

static dcethread_mutex bug_6386_mutex = DCETHREAD_MUTEX_INITIALIZER;

static void*
bug_6386_thread(void* data)
{
    MU_TRY_DCETHREAD(dcethread_mutex_lock(&bug_6386_mutex));
    MU_TRY_DCETHREAD(dcethread_mutex_unlock(&bug_6386_mutex));

    return data;
}

/* Test for regression of bug 6386, which causes
   deadlock when a thread interrupted during a
   dcethread_mutex_lock of a mutex held by
   the interrupting thread */
MU_TEST(dcethread_interrupt, bug_6386)
{
    dcethread* thread = NULL;

    MU_TRY_DCETHREAD(dcethread_mutex_lock(&bug_6386_mutex));
    MU_TRY_DCETHREAD(dcethread_create(&thread, NULL, bug_6386_thread, NULL));
    MU_TRY_DCETHREAD(dcethread_interrupt(thread));
    MU_TRY_DCETHREAD(dcethread_mutex_unlock(&bug_6386_mutex));
    MU_TRY_DCETHREAD(dcethread_join(thread, NULL));
}

#endif /* TEST */
