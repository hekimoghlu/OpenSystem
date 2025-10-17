/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
 * Copyright 2007 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/varargs.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define	DTRACEIOC	(('d' << 24) | ('t' << 16) | ('r' << 8))
#define	DTRACEIOC_MAX	17

void
fatal(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);

	fprintf(stderr, "%s: ", "badioctl");
	vfprintf(stderr, fmt, ap);

	if (fmt[strlen(fmt) - 1] != '\n')
		fprintf(stderr, ": %s\n", strerror(errno));

	exit(1);
}

void
badioctl(pid_t parent)
{
	int fd = -1, random, ps = sysconf(_SC_PAGESIZE);
	int i = 0, seconds;
	caddr_t addr;
	hrtime_t now, last = 0, end;

	if ((random = open("/dev/random", O_RDONLY)) == -1)
		fatal("couldn't open /dev/random");

	if ((addr = mmap(0, ps, PROT_READ | PROT_WRITE,
	    MAP_ANON | MAP_PRIVATE, -1, 0)) == (caddr_t)-1)
		fatal("mmap");

	for (;;) {
		unsigned int ioc;

		if ((now = gethrtime()) - last > NANOSEC) {
			if (kill(parent, 0) == -1 && errno == ESRCH) {
				/*
				 * Our parent died.  We will kill ourselves in
				 * sympathy.
				 */
				exit(0);
			}

			/*
			 * Once a second, we'll reopen the device.
			 */
			if (fd != -1)
				close(fd);

			fd = open("/dev/dtrace", O_RDONLY);

			if (fd == -1)
				fatal("couldn't open DTrace pseudo device");

			last = now;
		}


		if ((i++ % 1000) == 0) {
			/*
			 * Every thousand iterations, change our random gunk.
			 */
			read(random, addr, ps);
		}

		read(random, &ioc, sizeof (ioc));
		ioc %= DTRACEIOC_MAX;
		ioc++;
		ioctl(fd, DTRACEIOC | ioc, addr);
	}
}

int
main()
{
	pid_t child, parent = getpid();
	int status;

	for (;;) {
		if ((child = fork()) == 0)
			badioctl(parent);

		while (waitpid(child, &status, WEXITED) != child)
			continue;

		if (WIFEXITED(status)) {
			/*
			 * Our child exited by design -- we'll exit with
			 * the same status code.
			 */
			exit(WEXITSTATUS(status));
		}

		/*
		 * Our child died on a signal.  Respawn it.
		 */
		printf("badioctl: child died on signal %d; respawning.\n",
		    WTERMSIG(status));
		fflush(stdout);
	}

	/* NOTREACHED */
	return (0);
}
