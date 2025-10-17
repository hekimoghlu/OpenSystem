/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include "file.h"

#ifndef lint
FILE_RCSID("@(#)$File: cdf_time.c,v 1.19 2019/03/12 20:43:05 christos Exp $")
#endif

#include <time.h>
#ifdef TEST
#include <err.h>
#endif
#include <string.h>

#include "cdf.h"

#define isleap(y) ((((y) % 4) == 0) && \
    ((((y) % 100) != 0) || (((y) % 400) == 0)))

static const int mdays[] = {
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

/*
 * Return the number of days between jan 01 1601 and jan 01 of year.
 */
static int
cdf_getdays(int year)
{
	int days = 0;
	int y;

	for (y = CDF_BASE_YEAR; y < year; y++)
		days += isleap(y) + 365;

	return days;
}

/*
 * Return the day within the month
 */
static int
cdf_getday(int year, int days)
{
	size_t m;

	for (m = 0; m < __arraycount(mdays); m++) {
		int sub = mdays[m] + (m == 1 && isleap(year));
		if (days < sub)
			return days;
		days -= sub;
	}
	return days;
}

/*
 * Return the 0...11 month number.
 */
static int
cdf_getmonth(int year, int days)
{
	size_t m;

	for (m = 0; m < __arraycount(mdays); m++) {
		days -= mdays[m];
		if (m == 1 && isleap(year))
			days--;
		if (days <= 0)
			return CAST(int, m);
	}
	return CAST(int, m);
}

int
cdf_timestamp_to_timespec(struct timespec *ts, cdf_timestamp_t t)
{
	struct tm tm;
#ifdef HAVE_STRUCT_TM_TM_ZONE
	static char UTC[] = "UTC";
#endif
	int rdays;

	/* Unit is 100's of nanoseconds */
	ts->tv_nsec = (t % CDF_TIME_PREC) * 100;

	t /= CDF_TIME_PREC;
	tm.tm_sec = CAST(int, t % 60);
	t /= 60;

	tm.tm_min = CAST(int, t % 60);
	t /= 60;

	tm.tm_hour = CAST(int, t % 24);
	t /= 24;

	/* XXX: Approx */
	tm.tm_year = CAST(int, CDF_BASE_YEAR + (t / 365));

	rdays = cdf_getdays(tm.tm_year);
	t -= rdays - 1;
	tm.tm_mday = cdf_getday(tm.tm_year, CAST(int, t));
	tm.tm_mon = cdf_getmonth(tm.tm_year, CAST(int, t));
	tm.tm_wday = 0;
	tm.tm_yday = 0;
	tm.tm_isdst = 0;
#ifdef HAVE_STRUCT_TM_TM_GMTOFF
	tm.tm_gmtoff = 0;
#endif
#ifdef HAVE_STRUCT_TM_TM_ZONE
	tm.tm_zone = UTC;
#endif
	tm.tm_year -= 1900;
	ts->tv_sec = mktime(&tm);
	if (ts->tv_sec == -1) {
		errno = EINVAL;
		return -1;
	}
	return 0;
}

int
/*ARGSUSED*/
cdf_timespec_to_timestamp(cdf_timestamp_t *t, const struct timespec *ts)
{
#ifndef __lint__
	(void)&t;
	(void)&ts;
#endif
#ifdef notyet
	struct tm tm;
	if (gmtime_r(&ts->ts_sec, &tm) == NULL) {
		errno = EINVAL;
		return -1;
	}
	*t = (ts->ts_nsec / 100) * CDF_TIME_PREC;
	*t = tm.tm_sec;
	*t += tm.tm_min * 60;
	*t += tm.tm_hour * 60 * 60;
	*t += tm.tm_mday * 60 * 60 * 24;
#endif
	return 0;
}

char *
cdf_ctime(const time_t *sec, char *buf)
{
	char *ptr = ctime_r(sec, buf);
	if (ptr != NULL)
		return buf;
	(void)snprintf(buf, 26, "*Bad* %#16.16" INT64_T_FORMAT "x\n",
	    CAST(long long, *sec));
	return buf;
}


#ifdef TEST_TIME
int
main(int argc, char *argv[])
{
	struct timespec ts;
	char buf[25];
	static const cdf_timestamp_t tst = 0x01A5E403C2D59C00ULL;
	static const char *ref = "Sat Apr 23 01:30:00 1977";
	char *p, *q;

	cdf_timestamp_to_timespec(&ts, tst);
	p = cdf_ctime(&ts.tv_sec, buf);
	if ((q = strchr(p, '\n')) != NULL)
		*q = '\0';
	if (strcmp(ref, p) != 0)
		errx(1, "Error date %s != %s\n", ref, p);
	return 0;
}
#endif
