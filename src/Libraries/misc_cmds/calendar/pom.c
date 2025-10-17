/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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
 * Phase of the Moon.  Calculates the current phase of the moon.
 * Based on routines from `Practical Astronomy with Your Calculator',
 * by Duffett-Smith.  Comments give the section from the book that
 * particular piece of code was adapted from.
 *
 * -- Keith E. Brandt  VIII 1984
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sysexits.h>
#include <time.h>
#include <unistd.h> 

#include "calendar.h"

#ifndef	PI
#define	PI	  3.14159265358979323846
#endif
#define	EPOCH	  85
#define	EPSILONg  279.611371	/* solar ecliptic long at EPOCH */
#define	RHOg	  282.680403	/* solar ecliptic long of perigee at EPOCH */
#define	ECCEN	  0.01671542	/* solar orbit eccentricity */
#define	lzero	  18.251907	/* lunar mean long at EPOCH */
#define	Pzero	  192.917585	/* lunar mean long of perigee at EPOCH */
#define	Nzero	  55.204723	/* lunar mean long of node at EPOCH */
#define isleap(y) ((((y) % 4) == 0 && ((y) % 100) != 0) || ((y) % 400) == 0)

static void	adj360(double *);
static double	dtor(double);
static double	potm(double onday);
static double	potm_minute(double onday, int olddir);

void
pom(int year, double utcoffset, int *fms, int *nms)
{
	double ffms[MAXMOONS];
	double fnms[MAXMOONS];
	int i, j;

	fpom(year, utcoffset, ffms, fnms);

	j = 0;
	for (i = 0; ffms[i] != 0; i++)
		fms[j++] = round(ffms[i]);
	fms[i] = -1;
	for (i = 0; fnms[i] != 0; i++)
		nms[i] = round(fnms[i]);
	nms[i] = -1;
}

void
fpom(int year, double utcoffset, double *ffms, double *fnms)
{
	time_t tt;
	struct tm GMT, tmd_today, tmd_tomorrow;
	double days_today, days_tomorrow, today, tomorrow;
	int cnt, d;
	int yeardays;
	int olddir, newdir;
	double *pfnms, *pffms, t;

	pfnms = fnms;
	pffms = ffms;

	/*
	 * We take the phase of the moon one second before and one second
	 * after midnight.
	 */
	memset(&tmd_today, 0, sizeof(tmd_today));
	tmd_today.tm_year = year - 1900;
	tmd_today.tm_mon = 0;
	tmd_today.tm_mday = -1;		/* 31 December */
	tmd_today.tm_hour = 23;
	tmd_today.tm_min = 59;
	tmd_today.tm_sec = 59;
	memset(&tmd_tomorrow, 0, sizeof(tmd_tomorrow));
	tmd_tomorrow.tm_year = year - 1900;
	tmd_tomorrow.tm_mon = 0;
	tmd_tomorrow.tm_mday = 0;	/* 01 January */
	tmd_tomorrow.tm_hour = 0;
	tmd_tomorrow.tm_min = 0;
	tmd_tomorrow.tm_sec = 1;

	tt = mktime(&tmd_today);
	gmtime_r(&tt, &GMT);
	yeardays = 0;
	for (cnt = EPOCH; cnt < GMT.tm_year; ++cnt)
		yeardays += isleap(1900 + cnt) ? DAYSPERLEAPYEAR : DAYSPERYEAR;
	days_today = (GMT.tm_yday + 1) + ((GMT.tm_hour +
	    (GMT.tm_min / FSECSPERMINUTE) + (GMT.tm_sec / FSECSPERHOUR)) /
	    FHOURSPERDAY);
	days_today += yeardays;

	tt = mktime(&tmd_tomorrow);
	gmtime_r(&tt, &GMT);
	yeardays = 0;
	for (cnt = EPOCH; cnt < GMT.tm_year; ++cnt)
		yeardays += isleap(1900 + cnt) ? DAYSPERLEAPYEAR : DAYSPERYEAR;
	days_tomorrow = (GMT.tm_yday + 1) + ((GMT.tm_hour +
	    (GMT.tm_min / FSECSPERMINUTE) + (GMT.tm_sec / FSECSPERHOUR)) /
	    FHOURSPERDAY);
	days_tomorrow += yeardays;

	today = potm(days_today);		/* 30 December 23:59:59 */
	tomorrow = potm(days_tomorrow);		/* 31 December 00:00:01 */
	olddir = today > tomorrow ? -1 : +1;

	yeardays = 1 + (isleap(year) ? DAYSPERLEAPYEAR : DAYSPERYEAR); /* reuse */
	for (d = 0; d <= yeardays; d++) {
		today = potm(days_today);
		tomorrow = potm(days_tomorrow);
		newdir = today > tomorrow ? -1 : +1;
		if (olddir != newdir) {
			t = potm_minute(days_today - 1, olddir) +
			     utcoffset / FHOURSPERDAY;
			if (olddir == -1 && newdir == +1) {
				*pfnms = d - 1 + t;
				pfnms++;
			} else if (olddir == +1 && newdir == -1) {
				*pffms = d - 1 + t;
				pffms++;
			}
		}
		olddir = newdir;
		days_today++;
		days_tomorrow++;
	}
	*pffms = -1;
	*pfnms = -1;
}

static double
potm_minute(double onday, int olddir) {
	double period = FSECSPERDAY / 2.0;
	double p1, p2;
	double before, after;
	int newdir;

//	printf("---> days:%g olddir:%d\n", days, olddir);

	p1 = onday + (period / SECSPERDAY);
	period /= 2;

	while (period > 30) {	/* half a minute */
//		printf("period:%g - p1:%g - ", period, p1);
		p2 = p1 + (2.0 / SECSPERDAY);
		before = potm(p1);
		after = potm(p2);
//		printf("before:%10.10g - after:%10.10g\n", before, after);
		newdir = before < after ? -1 : +1;
		if (olddir != newdir)
			p1 += (period / SECSPERDAY);
		else
			p1 -= (period / SECSPERDAY);
		period /= 2;
//		printf("newdir:%d - p1:%10.10f - period:%g\n",
//		    newdir, p1, period);
	}
	p1 -= floor(p1);
	//exit(0);
	return (p1);
}

/*
 * potm --
 *	return phase of the moon, as a percentage [0 ... 100]
 */
static double
potm(double onday)
{
	double N, Msol, Ec, LambdaSol, l, Mm, Ev, Ac, A3, Mmprime;
	double A4, lprime, V, ldprime, D, Nm;

	N = 360 * onday / 365.2422;				/* sec 42 #3 */
	adj360(&N);
	Msol = N + EPSILONg - RHOg;				/* sec 42 #4 */
	adj360(&Msol);
	Ec = 360 / PI * ECCEN * sin(dtor(Msol));		/* sec 42 #5 */
	LambdaSol = N + Ec + EPSILONg;				/* sec 42 #6 */
	adj360(&LambdaSol);
	l = 13.1763966 * onday + lzero;				/* sec 61 #4 */
	adj360(&l);
	Mm = l - (0.1114041 * onday) - Pzero;			/* sec 61 #5 */
	adj360(&Mm);
	Nm = Nzero - (0.0529539 * onday);			/* sec 61 #6 */
	adj360(&Nm);
	Ev = 1.2739 * sin(dtor(2*(l - LambdaSol) - Mm));	/* sec 61 #7 */
	Ac = 0.1858 * sin(dtor(Msol));				/* sec 61 #8 */
	A3 = 0.37 * sin(dtor(Msol));
	Mmprime = Mm + Ev - Ac - A3;				/* sec 61 #9 */
	Ec = 6.2886 * sin(dtor(Mmprime));			/* sec 61 #10 */
	A4 = 0.214 * sin(dtor(2 * Mmprime));			/* sec 61 #11 */
	lprime = l + Ev + Ec - Ac + A4;				/* sec 61 #12 */
	V = 0.6583 * sin(dtor(2 * (lprime - LambdaSol)));	/* sec 61 #13 */
	ldprime = lprime + V;					/* sec 61 #14 */
	D = ldprime - LambdaSol;				/* sec 63 #2 */
	return(50 * (1 - cos(dtor(D))));			/* sec 63 #3 */
}

/*
 * dtor --
 *	convert degrees to radians
 */
static double
dtor(double deg)
{

	return(deg * PI / 180);
}

/*
 * adj360 --
 *	adjust value so 0 <= deg <= 360
 */
static void
adj360(double *deg)
{

	for (;;)
		if (*deg < 0)
			*deg += 360;
		else if (*deg > 360)
			*deg -= 360;
		else
			break;
}
