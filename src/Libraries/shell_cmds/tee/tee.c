/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#ifndef lint
static const char copyright[] =
"@(#) Copyright (c) 1988, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
#if 0
static char sccsid[] = "@(#)tee.c	8.1 (Berkeley) 6/6/93";
#endif
static const char rcsid[] =
  "$FreeBSD$";
#endif /* not lint */

#ifndef __APPLE__
#include <sys/capsicum.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct _list {
	struct _list *next;
	int fd;
	const char *name;
} LIST;
static LIST *head;

static void add(int, const char *);
static void usage(void);

int
main(int argc, char *argv[])
{
	LIST *p;
	int n, fd, rval, wval;
	char *bp;
	int append, ch, exitval;
	char *buf;
#define	BSIZE (8 * 1024)

	append = 0;
	while ((ch = getopt(argc, argv, "ai")) != -1)
		switch((char)ch) {
		case 'a':
			append = 1;
			break;
		case 'i':
			(void)signal(SIGINT, SIG_IGN);
			break;
		case '?':
		default:
			usage();
		}
	argv += optind;
	argc -= optind;

	if ((buf = malloc(BSIZE)) == NULL)
		err(1, "malloc");

#ifndef __APPLE__
	if (caph_limit_stdin() == -1 || caph_limit_stderr() == -1)
		err(EXIT_FAILURE, "unable to limit stdio");
#endif

	add(STDOUT_FILENO, "stdout");

	for (exitval = 0; *argv; ++argv)
		if ((fd = open(*argv, append ? O_WRONLY|O_CREAT|O_APPEND :
		    O_WRONLY|O_CREAT|O_TRUNC, DEFFILEMODE)) < 0) {
			warn("%s", *argv);
			exitval = 1;
		} else
			add(fd, *argv);

#ifndef __APPLE__
	if (caph_enter() < 0)
		err(EXIT_FAILURE, "unable to enter capability mode");
#endif
	while ((rval = read(STDIN_FILENO, buf, BSIZE)) > 0)
		for (p = head; p; p = p->next) {
			n = rval;
			bp = buf;
			do {
				if ((wval = write(p->fd, bp, n)) == -1) {
					warn("%s", p->name);
					exitval = 1;
					break;
				}
				bp += wval;
			} while (n -= wval);
		}
	if (rval < 0)
		err(1, "read");
	exit(exitval);
}

static void
usage(void)
{
	(void)fprintf(stderr, "usage: tee [-ai] [file ...]\n");
	exit(1);
}

static void
add(int fd, const char *name)
{
	LIST *p;
#ifndef __APPLE__
	cap_rights_t rights;

	if (fd == STDOUT_FILENO) {
		if (caph_limit_stdout() == -1)
			err(EXIT_FAILURE, "unable to limit stdout");
	} else {
		cap_rights_init(&rights, CAP_WRITE, CAP_FSTAT);
		if (caph_rights_limit(fd, &rights) < 0)
			err(EXIT_FAILURE, "unable to limit rights");
	}
#endif

	if ((p = malloc(sizeof(LIST))) == NULL)
		err(1, "malloc");
	p->fd = fd;
	p->name = name;
	p->next = head;
	head = p;
}
