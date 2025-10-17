/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

#ifdef API

void
dcethread_checkinterrupt(void)
{
    dcethread* thread = dcethread__self();
    int interruptible;
    int state;

    dcethread__lock(thread);
    state = thread->state;
    interruptible = thread->flag.interruptible;

    if (state == DCETHREAD_STATE_INTERRUPT && interruptible)
    {
        dcethread__change_state(thread, DCETHREAD_STATE_ACTIVE);
    }

    dcethread__unlock(thread);

    if (state == DCETHREAD_STATE_INTERRUPT && interruptible)
    {
        dcethread__dispatchinterrupt(thread);
    }

    return;
}

#endif

#ifdef TEST

#include "dcethread-test.h"

/* Test for regression of bug 6935,
   where dcethread_checkinterrupt() does not
   properly clear the interrupted state of the thread */
MU_TEST(dcethread_checkinterrupt, bug_6935)
{
    int interrupted_once = 0;
    int interrupted_twice = 0;

    dcethread_interrupt(dcethread_self());

    DCETHREAD_TRY
    {
        dcethread_checkinterrupt();
    }
    DCETHREAD_CATCH(dcethread_interrupt_e)
    {
        interrupted_once = 1;
    }
    DCETHREAD_ENDTRY;

    DCETHREAD_TRY
    {
        dcethread_checkinterrupt();
    }
    DCETHREAD_CATCH(dcethread_interrupt_e)
    {
        interrupted_twice = 1;
    }
    DCETHREAD_ENDTRY;

    MU_ASSERT(interrupted_once && !interrupted_twice);
}

#endif
