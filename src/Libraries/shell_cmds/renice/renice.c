/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <err.h>
#include <errno.h>
#include <limits.h>
#include <pwd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int	donice(int, int, int, bool);
static int	getnum(const char *, const char *, int *);
static void	usage(void);

/*
 * Change the priority (nice) of processes
 * or groups of processes which are already
 * running.
 */
int
main(int argc, char *argv[])
{
	struct passwd *pwd;
	bool havedelim = false, haveprio = false, incr = false;
	int errs = 0, prio = 0, who = 0, which = PRIO_PROCESS;

	for (argc--, argv++; argc > 0; argc--, argv++) {
		if (!havedelim) {
			/* can occur at any time prior to delimiter */
			if (strcmp(*argv, "-g") == 0) {
				which = PRIO_PGRP;
				continue;
			}
			if (strcmp(*argv, "-u") == 0) {
				which = PRIO_USER;
				continue;
			}
			if (strcmp(*argv, "-p") == 0) {
				which = PRIO_PROCESS;
				continue;
			}
			if (strcmp(*argv, "--") == 0) {
				havedelim = true;
				continue;
			}
			if (strcmp(*argv, "-n") == 0) {
				/* may occur only once, prior to priority */
				if (haveprio || incr || argc < 2)
					usage();
				incr = true;
				(void)argc--, argv++;
				/* fall through to priority */
			}
		}
		if (!haveprio) {
			/* must occur exactly once, prior to target */
			if (getnum("priority", *argv, &prio))
				return (1);
			haveprio = true;
			continue;
		}
		if (which == PRIO_USER) {
			if ((pwd = getpwnam(*argv)) != NULL)
				who = pwd->pw_uid;
			else if (getnum("uid", *argv, &who)) {
				errs++;
				continue;
			} else if (who < 0) {
				warnx("%s: bad value", *argv);
				errs++;
				continue;
			}
		} else {
			if (getnum("pid", *argv, &who)) {
				errs++;
				continue;
			}
			if (who < 0) {
				warnx("%s: bad value", *argv);
				errs++;
				continue;
			}
		}
		errs += donice(which, who, prio, incr);
	}
	if (!haveprio)
		usage();
	exit(errs != 0);
}

static int
donice(int which, int who, int prio, bool incr)
{
	int oldprio;

	errno = 0;
	oldprio = getpriority(which, who);
	if (oldprio == -1 && errno) {
		warn("%d: getpriority", who);
		return (1);
	}
	if (incr)
		prio = oldprio + prio;
	if (prio > PRIO_MAX)
		prio = PRIO_MAX;
	if (prio < PRIO_MIN)
		prio = PRIO_MIN;
	if (setpriority(which, who, prio) < 0) {
		warn("%d: setpriority", who);
		return (1);
	}
#ifndef __APPLE__
	fprintf(stderr, "%d: old priority %d, new priority %d\n", who,
	    oldprio, prio);
#endif
	return (0);
}

static int
getnum(const char *com, const char *str, int *val)
{
	long v;
	char *ep;

	errno = 0;
	v = strtol(str, &ep, 10);
	if (v < INT_MIN || v > INT_MAX || errno == ERANGE) {
		warnx("%s argument %s is out of range.", com, str);
		return (1);
	}
	if (ep == str || *ep != '\0' || errno != 0) {
		warnx("%s argument %s is invalid.", com, str);
		return (1);
	}

	*val = (int)v;
	return (0);
}

static void
usage(void)
{
	fprintf(stderr, "%s\n%s\n",
"usage: renice priority [[-p] pid ...] [[-g] pgrp ...] [[-u] user ...]",
"       renice -n increment [[-p] pid ...] [[-g] pgrp ...] [[-u] user ...]");
	exit(1);
}
