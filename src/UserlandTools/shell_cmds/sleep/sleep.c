/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <err.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t report_requested;

static void
report_request(int signo __unused)
{
	report_requested = 1;
}

static void __dead2
usage(void)
{
	fprintf(stderr, "usage: sleep number[unit] [...]\n"
	    "Unit can be 's' (seconds, the default), "
	    "m (minutes), h (hours), or d (days).\n");
	exit(1);
}

static double
parse_interval(const char *arg)
{
	double num;
	char unit, extra;

	switch (sscanf(arg, "%lf%c%c", &num, &unit, &extra)) {
	case 2:
		switch (unit) {
		case 'd':
			num *= 24;
			/* FALLTHROUGH */
		case 'h':
			num *= 60;
			/* FALLTHROUGH */
		case 'm':
			num *= 60;
			/* FALLTHROUGH */
		case 's':
			if (!isnan(num))
				return (num);
		}
		break;
	case 1:
		if (!isnan(num))
			return (num);
	}
	warnx("invalid time interval: %s", arg);
	return (INFINITY);
}

int
main(int argc, char *argv[])
{
	struct timespec time_to_sleep;
	double seconds;
	time_t original;

#ifndef __APPLE__
	if (caph_limit_stdio() < 0 || caph_enter() < 0)
		err(1, "capsicum");
#endif

	while (getopt(argc, argv, "") != -1)
		usage();
	argc -= optind;
	argv += optind;
	if (argc < 1)
		usage();

	seconds = 0;
	while (argc--)
		seconds += parse_interval(*argv++);
	if (seconds > INT_MAX)
		usage();
	if (seconds < 1e-9)
		exit(0);
	original = time_to_sleep.tv_sec = (time_t)seconds;
	time_to_sleep.tv_nsec = 1e9 * (seconds - time_to_sleep.tv_sec);

	signal(SIGINFO, report_request);

	while (nanosleep(&time_to_sleep, &time_to_sleep) != 0) {
		if (errno != EINTR)
			err(1, "nanosleep");
		if (report_requested) {
			/* Reporting does not bother with nanoseconds. */
			warnx("about %ld second(s) left out of the original %ld",
			    (long)time_to_sleep.tv_sec, (long)original);
			report_requested = 0;
		}
	}

	exit(0);
}
