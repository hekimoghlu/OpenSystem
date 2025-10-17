/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

#include "sudo.h"

static volatile sig_atomic_t got_sigttou;

/*
 * SIGTTOU signal handler for tcsetpgrp_nobg() that just sets a flag.
 */
static void
sigttou(int signo)
{
    got_sigttou = 1;
}

/*
 * Like tcsetpgrp() but restarts on EINTR _except_ for SIGTTOU.
 * Returns 0 on success or -1 on failure, setting errno.
 * Sets got_sigttou on failure if interrupted by SIGTTOU.
 */
int
tcsetpgrp_nobg(int fd, pid_t pgrp_id)
{
    struct sigaction sa, osa;
    int rc;

    /*
     * If we receive SIGTTOU from tcsetpgrp() it means we are
     * not in the foreground process group.
     * This avoid a TOCTOU race compared to using tcgetpgrp().
     */
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* do not restart syscalls */
    sa.sa_handler = sigttou;
    got_sigttou = 0;
    (void)sigaction(SIGTTOU, &sa, &osa);
    do {
	rc = tcsetpgrp(fd, pgrp_id);
    } while (rc != 0 && errno == EINTR && !got_sigttou);
    (void)sigaction(SIGTTOU, &osa, NULL);

    return rc;
}
