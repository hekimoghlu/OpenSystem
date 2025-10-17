/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#include "dcethread-exception.h"

#ifdef API

void
dcethread_exit(void* status)
{
    dcethread* thread = dcethread__self();

    dcethread__lock(thread);
    thread->status = status;
    dcethread__unlock(thread);

    dcethread__exc_raise(&dcethread_exit_thread_e, __FILE__, __LINE__);
}

#endif /* API */

#ifdef TEST

#include "dcethread-test.h"

static volatile int finally_reached = 0;

static void*
basic_thread(void* arg)
{
    DCETHREAD_TRY
    {
        dcethread_exit(arg);
    }
    DCETHREAD_FINALLY
    {
        MU_ASSERT(!finally_reached);
        finally_reached = 1;
    }
    DCETHREAD_ENDTRY;

    return NULL;
}

MU_TEST(dcethread_exit, basic)
{
    dcethread* thread;
    void* status;

    MU_TRY_DCETHREAD( dcethread_create(&thread, NULL, basic_thread, (void*) 0xDEADBEEF) );
    MU_TRY_DCETHREAD( dcethread_join(thread, &status) );

    MU_ASSERT(finally_reached);
    MU_ASSERT_EQUAL(MU_TYPE_POINTER, status, (void*) 0xDEADBEEF);
}

#endif /* TEST */
