/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
#include <dce/dcethread.h>
#include <signal.h>
#include <errno.h>
#include <config.h>

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

static void *async_signal_handler (void *dummy);

/* FIXME: do we need to retain thread? */
static dcethread* thread_to_interrupt;
static sigset_t async_sigset;
static pthread_t helper_thread = (pthread_t)0;

/* -------------------------------------------------------------------- */

/*
 * A S Y N C _ S I G N A L _ H A N D L E R
 *
 * This async signal handler is already running on the correct thread
 * stack.   ALL async signals map to a "cancel".  A cancel unwind happens
 * only at well defined points, so we can't RAISE the exception; just
 * post the cancel.
 */
static void *async_signal_handler(void *dummy ATTRIBUTE_UNUSED)
{
    /*
     * Wait for and handle asynchronous signals.
     */
    while (1)
    {
        int sig;

        sigwait(&async_sigset, &sig);
	dcethread__lock(thread_to_interrupt);
	dcethread__interrupt(thread_to_interrupt);
	dcethread__unlock(thread_to_interrupt);
    }

    return NULL;
}

/*
 * P T H R E A D _ S I G N A L _ T O _ C A N C E L _ N P
 *
 * Async signal handling consists of creating an exception package helper
 * thread.  This helper thread will sigwait() on all async signals of
 * interest and will convert specific asynchronous signals (defined in the
 * async signal handler) to exceptions.  The helper thread then posts the
 * cancel to the thread that initialized the exception package for
 * 'handling'.
 */
void
dcethread_signal_to_interrupt(sigset_t *asigset, dcethread* thread)
{
    /*
     * The helper thread will need the thread id of the first thread
     * to initialize the exception package.  The thread that initialized
     * the exception package will receive a cancel when an asynchronous
     * signal is received.
     */
    thread_to_interrupt = thread;
    async_sigset = *asigset;

    dcethread_lock_global();

    if (helper_thread != (pthread_t)0)
    {
        pthread_cancel(helper_thread);
        pthread_detach(helper_thread);
    }

    /*
     * Create a 'helper thread' to catch aynchronous signals.
     */
    pthread_create(&helper_thread, NULL, async_signal_handler, 0);

    /*
     * The 'helper thread' will never be joined so toss any pthread package
     * internal record of the thread away to conserve resources.
     */
    pthread_detach(helper_thread);

    dcethread_unlock_global();
}
