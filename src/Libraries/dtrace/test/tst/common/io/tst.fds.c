/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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

#include <assert.h>
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>

#include <sys/ioctl.h>

static sigjmp_buf env;

static void
interrupt(int sig)
{
	siglongjmp(env, sig);
}

int
main(int argc, char *argv[])
{
	const char *file = "/etc/hosts";
	int i, n, fds[10];
	struct sigaction act;

	if (argc > 1) {
		(void) fprintf(stderr, "Usage: %s\n", argv[0]);
		return (EXIT_FAILURE);
	}

	act.sa_handler = interrupt;
	act.sa_flags = 0;

	(void) sigemptyset(&act.sa_mask);
	(void) sigaction(SIGUSR1, &act, NULL);

	n = getdtablesize();
	for (i = 0; i < n; ++i) close(i);
	n = 0;

	/*
	 * With all of our file descriptors closed, wait here spinning in bogus
	 * ioctl() calls until DTrace hits us with a SIGUSR1 to start the test.
	 */
	if (sigsetjmp(env, 1) == 0) {
		for (;;)
			(void) ioctl(-1, -1, NULL);
	}

	/*
	 * To test the fds[] array, we open /dev/null (a file with reliable
	 * pathname and properties) using various flags and seek offsets.
	 */
	fds[n++] = open(file, O_RDONLY);
	fds[n++] = open(file, O_WRONLY);
	fds[n++] = open(file, O_RDWR);

	fds[n++] = open(file, O_RDWR | O_APPEND | O_CREAT | O_ASYNC |
	    O_NOCTTY | O_NONBLOCK | O_NDELAY | O_SYNC );

	fds[n++] = open(file, O_RDWR);
	(void) lseek(fds[n - 1], 123, SEEK_SET);

	/*
	 * Once we have all the file descriptors in the state we want to test,
	 * issue a bogus ioctl() on each fd with cmd -1 and arg NULL to whack
	 * our DTrace script into recording the content of the fds[] array.
	 */
	for (i = 0; i < n; i++)
		(void) ioctl(fds[i], -1, NULL);

	assert(n <= sizeof (fds) / sizeof (fds[0]));
	exit(0);
}
