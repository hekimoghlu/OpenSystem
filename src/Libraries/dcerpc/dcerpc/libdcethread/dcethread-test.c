/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include <stdio.h>

#include "dcethread-test.h"
#include "dcethread-debug.h"
#include "dcethread-private.h"
#include "dcethread-exception.h"

static void
debug_callback(const char* file, unsigned int line, int level, const char* message, void* data)
{
    int mu_level = MU_LEVEL_WARNING;

    switch (level)
    {
    case DCETHREAD_DEBUG_ERROR:
	Mu_Interface_Result(file, line, MU_STATUS_FAILURE, "%s", message);
        return;
    case DCETHREAD_DEBUG_WARNING:
	mu_level = MU_LEVEL_WARNING;
	break;
    case DCETHREAD_DEBUG_INFO:
	mu_level = MU_LEVEL_INFO;
	break;
    case DCETHREAD_DEBUG_VERBOSE:
	mu_level = MU_LEVEL_VERBOSE;
	break;
    case DCETHREAD_DEBUG_TRACE:
	mu_level = MU_LEVEL_TRACE;
	break;
    }

    Mu_Interface_Event(file, line, mu_level, "%s", message);
}

static void
uncaught_callback(dcethread_exc* exc, const char* file, unsigned int line, void* data)
{
    if (!dcethread__exc_matches(exc, &dcethread_interrupt_e) &&
        !dcethread__exc_matches(exc, &dcethread_exit_thread_e))
    {
        const char* name = dcethread__exc_getname(exc);
        if (name)
        {
            Mu_Interface_Result(file, line, MU_STATUS_EXCEPTION, "exception %s thrown but not caught", name);
        }
        else
        {
            Mu_Interface_Result(file, line, MU_STATUS_EXCEPTION, "exception %i thrown but not caught", dcethread__exc_getstatus(exc));
        }
    }

    pthread_exit(0);
}

void
dcethread__test_init()
{
    dcethread__debug_set_callback(debug_callback, NULL);
    dcethread__exc_set_uncaught_handler(uncaught_callback, NULL);
}

MU_LIBRARY_SETUP()
{
    dcethread__test_init();
}
