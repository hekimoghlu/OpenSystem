/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
"@(#) Copyright (c) 1980, 1988, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
#if 0
static char sccsid[] = "@(#)leave.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <err.h>
#include <ctype.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static void doalarm(u_int);
static void usage(void);

/*
 * leave [[+]hhmm]
 *
 * Reminds you when you have to leave.
 * Leave prompts for input and goes away if you hit return.
 * It nags you like a mother hen.
 */
int
main(int argc, char **argv)
{
	u_int secs;
	int hours, minutes;
	char c, *cp = NULL;
	struct tm *t;
	time_t now;
	int plusnow, t_12_hour;
	char buf[50];

	if (setlocale(LC_TIME, "") == NULL)
		warn("setlocale");

	if (argc < 2) {
#define	MSG1	"When do you have to leave? "
		(void)write(STDOUT_FILENO, MSG1, sizeof(MSG1) - 1);
		cp = fgets(buf, sizeof(buf), stdin);
		if (cp == NULL || *cp == '\n')
			exit(0);
	} else if (argc > 2)
		usage();
	else
		cp = argv[1];

	if (*cp == '+') {
		plusnow = 1;
		++cp;
	} else
		plusnow = 0;

	for (hours = 0; (c = *cp) && c != '\n'; ++cp) {
		if (!isdigit(c))
			usage();
		hours = hours * 10 + (c - '0');
	}
	minutes = hours % 100;
	hours /= 100;

	if (minutes < 0 || minutes > 59)
		usage();
	if (plusnow)
		secs = hours * 60 * 60 + minutes * 60;
	else {
		(void)time(&now);
		t = localtime(&now);

		if (hours > 23)
			usage();

		/* Convert tol to 12 hr time (0:00...11:59) */
		if (hours > 11)
			hours -= 12;

		/* Convert tm to 12 hr time (0:00...11:59) */
		if (t->tm_hour > 11)
			t_12_hour = t->tm_hour - 12;
		else
			t_12_hour = t->tm_hour;

		if (hours < t_12_hour ||
	 	   (hours == t_12_hour && minutes <= t->tm_min))
			/* Leave time is in the past so we add 12 hrs */
			hours += 12;

		secs = (hours - t_12_hour) * 60 * 60;
		secs += (minutes - t->tm_min) * 60;
		secs -= now % 60;	/* truncate (now + secs) to min */
	}
	doalarm(secs);
	exit(0);
}

void
doalarm(u_int secs)
{
	int bother;
	time_t daytime;
	char tb[80];
	int pid;

	if ((pid = fork())) {
		(void)time(&daytime);
		daytime += secs;
		strftime(tb, sizeof(tb), "%+", localtime(&daytime));
		printf("Alarm set for %s. (pid %d)\n", tb, pid);
		exit(0);
	}
	sleep((u_int)2);		/* let parent print set message */
	if (secs >= 2)
		secs -= 2;

	/*
	 * if write fails, we've lost the terminal through someone else
	 * causing a vhangup by logging in.
	 */
#define	FIVEMIN	(5 * 60)
#define	MSG2	"\07\07You have to leave in 5 minutes.\n"
	if (secs >= FIVEMIN) {
		sleep(secs - FIVEMIN);
		if (write(STDOUT_FILENO, MSG2, sizeof(MSG2) - 1) != sizeof(MSG2) - 1)
			exit(0);
		secs = FIVEMIN;
	}

#define	ONEMIN	(60)
#define	MSG3	"\07\07Just one more minute!\n"
	if (secs >= ONEMIN) {
		sleep(secs - ONEMIN);
		if (write(STDOUT_FILENO, MSG3, sizeof(MSG3) - 1) != sizeof(MSG3) - 1)
			exit(0);
	}

#define	MSG4	"\07\07Time to leave!\n"
	for (bother = 10; bother--;) {
		sleep((u_int)ONEMIN);
		if (write(STDOUT_FILENO, MSG4, sizeof(MSG4) - 1) != sizeof(MSG4) - 1)
			exit(0);
	}

#define	MSG5	"\07\07That was the last time I'll tell you.  Bye.\n"
	(void)write(STDOUT_FILENO, MSG5, sizeof(MSG5) - 1);
	exit(0);
}

static void
usage(void)
{
	fprintf(stderr, "usage: leave [[+]hhmm]\n");
	exit(1);
}
