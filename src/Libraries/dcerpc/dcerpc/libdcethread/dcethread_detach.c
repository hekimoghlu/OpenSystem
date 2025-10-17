/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
dcethread_detach(dcethread *thread)
{
    if (!thread->flag.joinable)
    {
        DCETHREAD_WARNING("Detaching implicit dcethread %p is ill-advised", thread);
    }

    dcethread__lock(thread);
    dcethread__release(thread);
    dcethread__unlock(thread);

    return 0;
}

int
dcethread_detach_throw(dcethread *thread)
{
    DCETHREAD_WRAP_THROW(dcethread_detach(thread));
}

#endif /* API */

#ifdef TEST

#include "dcethread-test.h"

static void*
basic_thread(void* data)
{
    return NULL;
}

MU_TEST(dcethread_detach, basic)
{
    dcethread* thread;

    MU_TRY_DCETHREAD( dcethread_create(&thread, NULL, basic_thread, NULL ) );
    MU_TRY_DCETHREAD( dcethread_detach(thread) );
}

#endif /* TEST */
