/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
#include <unistd.h>

/* Utility library. */

#include <killme_after.h>

/* killme_after - self-assured death */

void    killme_after(unsigned int seconds)
{
    struct sigaction sig_action;

    /*
     * Schedule an ALARM signal, and make sure the signal will be delivered
     * even if we are being called from a signal handler and SIGALRM delivery
     * is blocked.
     */
    alarm(0);
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_flags = 0;
    sig_action.sa_handler = SIG_DFL;
    sigaction(SIGALRM, &sig_action, (struct sigaction *) 0);
    alarm(seconds);
    sigaddset(&sig_action.sa_mask, SIGALRM);
    sigprocmask(SIG_UNBLOCK, &sig_action.sa_mask, (sigset_t *) 0);
}
