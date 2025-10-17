/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include <sys/cdefs.h>

#ifdef __APPLE__
#include <mach/machine/vm_param.h>
#endif

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

static int
exec_shortargs(char *argv[])
{
	char *flag_arg = argv[2];
	char *sentinel = argv[3];
	char * nargv[] = { argv[0], __DECONST(char *, "--spin"), flag_arg,
	    sentinel, NULL };
	char * const nenvp[] = { NULL };

	execve(argv[0], nargv, nenvp);
	err(1, "execve");
}

static int
exec_largeargs(char *argv[])
{
	char *flag_arg = argv[2];
	char *sentinel = argv[3];
	/*
	 * Account for each argument and their NUL terminator, as well as an
	 * extra NUL terminator.
	 */
	size_t bufsz = ARG_MAX -
#ifdef __APPLE__
	    /*
	     * Knock a couple pages off to account for various bits that get
	     * accounted towards ARG_MAX; it still tests for a much higher
	     * argument limit than pgrep/pkill previously had.
	     */
	    (PAGE_SIZE * 4) -
#endif
	    ((strlen(argv[0]) + 1) + sizeof("--spin") + (strlen(flag_arg) + 1) +
	    (strlen(sentinel) + 1) + 1);
	char *s = NULL;
	char * nargv[] = { argv[0], __DECONST(char *, "--spin"), flag_arg, NULL,
	    sentinel, NULL };
	char * const nenvp[] = { NULL };

	/*
	 * Our heuristic may or may not be accurate, we'll keep trying with
	 * smaller argument sizes as needed until we stop getting E2BIG.
	 */
	do {
		if (s == NULL)
			s = malloc(bufsz + 1);
		else
			s = realloc(s, bufsz + 1);
		if (s == NULL)
			abort();
		memset(s, 'x', bufsz);
		s[bufsz] = '\0';
		nargv[3] = s;

		execve(argv[0], nargv, nenvp);
		bufsz--;
	} while (errno == E2BIG);
	err(1, "execve");
}

int
main(int argc, char *argv[])
{

	if (argc > 1 && strcmp(argv[1], "--spin") == 0) {
		int fd;

		if (argc < 4) {
			fprintf(stderr, "usage: %s --spin flagfile ...\n", argv[0]);
			return (1);
		}

		fd = open(argv[2], O_RDWR | O_CREAT, 0755);
		if (fd < 0)
			err(1, "%s", argv[2]);
		close(fd);

		for (;;) {
			sleep(1);
		}

		return (1);
	}

	if (argc != 4) {
		fprintf(stderr, "usage: %s [--short | --long] flagfile sentinel\n",
		    argv[0]);
		return (1);
	}

	if (strcmp(argv[1], "--short") == 0)
		exec_shortargs(argv);
	else
		exec_largeargs(argv);
}
