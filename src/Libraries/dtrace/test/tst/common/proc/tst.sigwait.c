/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
 * Copyright 2006 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include <signal.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

void
main(int argc, char **argv)
{
	struct sigevent ev;
	struct itimerspec ts;
	sigset_t set;
	timer_t tid;
	char *cmd = argv[0];

	ev.sigev_notify = SIGEV_SIGNAL;
	ev.sigev_signo = SIGUSR1;

	if (timer_create(CLOCK_REALTIME, &ev, &tid) == -1) {
		(void) fprintf(stderr, "%s: cannot create CLOCK_HIGHRES "
		    "timer: %s\n", cmd, strerror(errno));
		exit(EXIT_FAILURE);
	}

	(void) sigemptyset(&set);
	(void) sigaddset(&set, SIGUSR1);
	(void) sigprocmask(SIG_BLOCK, &set, NULL);

	ts.it_value.tv_sec = 1;
	ts.it_value.tv_nsec = 0;
	ts.it_interval.tv_sec = 0;
	ts.it_interval.tv_nsec = NANOSEC / 2;

	if (timer_settime(tid, TIMER_RELTIME, &ts, NULL) == -1) {
		(void) fprintf(stderr, "%s: timer_settime() failed: %s\n",
		    cmd, strerror(errno));
		exit(EXIT_FAILURE);
	}

	for (;;) {
		(void) sigwait(&set);
	}
}
