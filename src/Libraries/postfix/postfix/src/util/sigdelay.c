/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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

#include <sys_defs.h>
#include <signal.h>

/* Utility library. */

#include "msg.h"
#include "posix_signals.h"
#include "sigdelay.h"

/* Application-specific. */

static sigset_t saved_sigmask;
static sigset_t block_sigmask;
static int suspending;
static int siginit_done;

/* siginit - compute signal mask only once */

static void siginit(void)
{
    int     sig;

    siginit_done = 1;
    sigemptyset(&block_sigmask);
    for (sig = 1; sig < NSIG; sig++)
	sigaddset(&block_sigmask, sig);
}

/* sigresume - deliver delayed signals and disable signal delay */

void    sigresume(void)
{
    if (suspending != 0) {
	suspending = 0;
	if (sigprocmask(SIG_SETMASK, &saved_sigmask, (sigset_t *) 0) < 0)
	    msg_fatal("sigresume: sigprocmask: %m");
    }
}

/* sigdelay - save signal mask and block all signals */

void    sigdelay(void)
{
    if (siginit_done == 0)
	siginit();
    if (suspending == 0) {
	suspending = 1;
	if (sigprocmask(SIG_BLOCK, &block_sigmask, &saved_sigmask) < 0)
	    msg_fatal("sigdelay: sigprocmask: %m");
    }
}

#ifdef TEST

 /*
  * Test program - press Ctrl-C twice while signal delivery is delayed, and
  * see how many signals are delivered when signal delivery is resumed.
  */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

static void gotsig(int sig)
{
    printf("Got signal %d\n", sig);
}

int     main(int unused_argc, char **unused_argv)
{
    signal(SIGINT, gotsig);
    signal(SIGQUIT, gotsig);

    printf("Delaying signal delivery\n");
    sigdelay();
    sleep(5);
    printf("Resuming signal delivery\n");
    sigresume();
    exit(0);
}

#endif
