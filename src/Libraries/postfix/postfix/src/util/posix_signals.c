/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
/* System library. */

#include "sys_defs.h"
#include <signal.h>
#include <errno.h>

/* Utility library.*/

#include "posix_signals.h"

#ifdef MISSING_SIGSET_T

int     sigemptyset(sigset_t *m)
{
    return *m = 0;
}

int     sigaddset(sigset_t *set, int signum)
{
    *set |= sigmask(signum);
    return 0;
}

int     sigprocmask(int how, sigset_t *set, sigset_t *old)
{
    int previous;

    if (how == SIG_BLOCK)
	previous = sigblock(*set);
    else if (how == SIG_SETMASK)
	previous = sigsetmask(*set);
    else if (how == SIG_UNBLOCK) {
	int     m = sigblock(0);

	previous = sigsetmask(m & ~*set);
    } else {
	errno = EINVAL;
	return -1;
    }

    if (old)
	*old = previous;
    return 0;
}

#endif

#ifdef MISSING_SIGACTION

static struct sigaction actions[NSIG] = {};

static int sighandle(int signum)
{
    if (signum == SIGCHLD) {
	/* XXX If the child is just stopped, don't invoke the handler.	 */
    }
    actions[signum].sa_handler(signum);
}

int     sigaction(int sig, struct sigaction *act, struct sigaction *oact)
{
    static int initialized = 0;

    if (!initialized) {
	int     i;

	for (i = 0; i < NSIG; i++)
	    actions[i].sa_handler = SIG_DFL;
	initialized = 1;
    }
    if (sig <= 0 || sig >= NSIG) {
	errno = EINVAL;
	return -1;
    }
    if (oact)
	*oact = actions[sig];

    {
	struct sigvec mine = {
	    sighandle, act->sa_mask,
	    act->sa_flags & SA_RESTART ? SV_INTERRUPT : 0
	};

	if (sigvec(sig, &mine, NULL))
	    return -1;
    }

    actions[sig] = *act;
    return 0;
}

#endif
