/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#include <err.h>
#include <errno.h>
#include <locale.h>
#ifndef __APPLE__
#include <login_cap.h>
#endif
#include <langinfo.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "calendar.h"

#define	UTCOFFSET_NOTSET	100	/* Expected between -24 and +24 */
#define	LONGITUDE_NOTSET	1000	/* Expected between -360 and +360 */

struct passwd	*pw;
int		doall = 0;
int		debug = 0;
static char	*DEBUG = NULL;
static time_t	f_time = 0;
double		UTCOffset = UTCOFFSET_NOTSET;
int		EastLongitude = LONGITUDE_NOTSET;
#ifdef WITH_ICONV
const char	*outputEncoding = NULL;
#endif

static void	usage(void) __dead2;

int
main(int argc, char *argv[])
{
#ifdef __APPLE__
	/*
	 * rdar://problem/5900695: calendar(1) documents taking the weekend into
	 * account in the default case, which would imply a default -A 1.
	 */
	int	f_dayAfter = 1;		/* days after current date */
#else
	int	f_dayAfter = 0;		/* days after current date */
#endif
	int	f_dayBefore = 0;	/* days before current date */
	int	Friday = 5;		/* day before weekend */

	int ch;
	struct tm tp1, tp2;

	(void)setlocale(LC_ALL, "");

	while ((ch = getopt(argc, argv, "-A:aB:D:dF:f:l:t:U:W:?")) != -1)
		switch (ch) {
		case '-':		/* backward contemptible */
		case 'a':
			if (getuid()) {
				errno = EPERM;
				err(1, NULL);
			}
			doall = 1;
			break;

		case 'W': /* we don't need no steenking Fridays */
			Friday = -1;
			/* FALLTHROUGH */

		case 'A': /* days after current date */
			f_dayAfter = atoi(optarg);
			if (f_dayAfter < 0)
				errx(1, "number of days must be positive");
			break;

		case 'B': /* days before current date */
			f_dayBefore = atoi(optarg);
			if (f_dayBefore < 0)
				errx(1, "number of days must be positive");
			break;

		case 'D': /* debug output of sun and moon info */
			DEBUG = optarg;
			break;

		case 'd': /* debug output of current date */
			debug = 1;
			break;

		case 'F': /* Change the time: When does weekend start? */
			Friday = atoi(optarg);
			break;

		case 'f': /* other calendar file */
			calendarFile = optarg;
			break;

		case 'l': /* Change longitudal position */
			EastLongitude = strtol(optarg, NULL, 10);
			break;

		case 't': /* other date, for tests */
			f_time = Mktime(optarg);
			break;

		case 'U': /* Change UTC offset */
			UTCOffset = strtod(optarg, NULL);
			break;

		case '?':
		default:
			usage();
		}

	argc -= optind;
	argv += optind;

	if (argc)
		usage();

	/* use current time */
	if (f_time <= 0)
		(void)time(&f_time);

	/* if not set, determine where I could be */
	{
		if (UTCOffset == UTCOFFSET_NOTSET &&
		    EastLongitude == LONGITUDE_NOTSET) {
			/* Calculate on difference between here and UTC */
			time_t t;
			struct tm tm;
			long utcoffset, hh, mm, ss;
			double uo;

			time(&t);
			localtime_r(&t, &tm);
			utcoffset = tm.tm_gmtoff;
			/* seconds -> hh:mm:ss */
			hh = utcoffset / SECSPERHOUR;
			utcoffset %= SECSPERHOUR;
			mm = utcoffset / SECSPERMINUTE;
			utcoffset %= SECSPERMINUTE;
			ss = utcoffset;

			/* hh:mm:ss -> hh.mmss */
			uo = mm + (100.0 * (ss / 60.0));
			uo /=  60.0 / 100.0;
			uo = hh + uo / 100;

			UTCOffset = uo;
			EastLongitude = UTCOffset * 15;
		} else if (UTCOffset == UTCOFFSET_NOTSET) {
			/* Base on information given */
			UTCOffset = EastLongitude / 15;
		} else if (EastLongitude == LONGITUDE_NOTSET) {
			/* Base on information given */
			EastLongitude = UTCOffset * 15;
		}
	}

	settimes(f_time, f_dayBefore, f_dayAfter, Friday, &tp1, &tp2);
	generatedates(&tp1, &tp2);

	/*
	 * FROM now on, we are working in UTC.
	 * This will only affect moon and sun related events anyway.
	 */
	if (setenv("TZ", "UTC", 1) != 0)
		errx(1, "setenv: %s", strerror(errno));
	tzset();

	if (debug)
		dumpdates();

	if (DEBUG != NULL) {
		dodebug(DEBUG);
		exit(0);
	}

	if (doall)
		while ((pw = getpwent()) != NULL) {
			pid_t pid;

			if (chdir(pw->pw_dir) == -1)
				continue;
			pid = fork();
			if (pid < 0)
				err(1, "fork");
			if (pid == 0) {
#ifdef __APPLE__
				(void)setegid(pw->pw_gid);
				(void)initgroups(pw->pw_name, pw->pw_gid);
				(void)seteuid(pw->pw_uid);
#else
				login_cap_t *lc;

				lc = login_getpwclass(pw);
				if (setusercontext(lc, pw, pw->pw_uid,
				    LOGIN_SETALL) != 0)
					errx(1, "setusercontext");
#endif
				setenv("HOME", pw->pw_dir, 1);
				cal();
				exit(0);
			}
		}
	else {
#ifdef WITH_ICONV
		/* Save the information about the encoding used in the terminal. */
		outputEncoding = strdup(nl_langinfo(CODESET));
		if (outputEncoding == NULL)
			errx(1, "cannot allocate memory");
#endif
		cal();
	}
	exit(0);
}


static void __dead2
usage(void)
{

	fprintf(stderr, "%s\n%s\n%s\n",
	    "usage: calendar [-A days] [-a] [-B days] [-D sun|moon] [-d]",
	    "		     [-F friday] [-f calendarfile] [-l longitude]",
	    "		     [-t dd[.mm[.year]]] [-U utcoffset] [-W days]"
	    );
	exit(1);
}
