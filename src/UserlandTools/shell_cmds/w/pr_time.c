/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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

#ifndef __APPLE__
__FBSDID("$FreeBSD$");

#ifndef lint
static const char sccsid[] = "@(#)pr_time.c	8.2 (Berkeley) 4/4/94";
#endif
#endif

#include <sys/types.h>
#include <sys/time.h>

#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <libxo/xo.h>

#include "extern.h"

/*
 * pr_attime --
 *	Print the time since the user logged in.
 */
int
pr_attime(time_t *started, time_t *now)
{
	static wchar_t buf[256];
	struct tm tp, tm;
	time_t diff;
	const wchar_t *fmt;
	int len, width, offset = 0;

	tp = *localtime(started);
	tm = *localtime(now);
	diff = *now - *started;

	/* If more than a week, use day-month-year. */
	if (diff > 86400 * 7)
		fmt = L"%d%b%y";

	/* If not today, use day-hour-am/pm. */
	else if (tm.tm_mday != tp.tm_mday ||
		 tm.tm_mon != tp.tm_mon ||
		 tm.tm_year != tp.tm_year) {
	/* The line below does not take DST into consideration */
	/* else if (*now / 86400 != *started / 86400) { */
		fmt = use_ampm ? L"%a%I%p" : L"%a%H";
	}

	/* Default is hh:mm{am,pm}. */
	else {
		fmt = use_ampm ? L"%l:%M%p" : L"%k:%M";
	}

	(void)wcsftime(buf, sizeof(buf), fmt, &tp);
	len = wcslen(buf);
	width = wcswidth(buf, len);
	xo_attr("since", "%lu", (unsigned long) *started);
	xo_attr("delta", "%lu", (unsigned long) diff);
	if (len == width)
		xo_emit("{:login-time/%-7.7ls/%ls}", buf);
	else if (width < 7)
		xo_emit("{:login-time/%ls}%.*s", buf, 7 - width, "      ");
	else {
		xo_emit("{:login-time/%ls}", buf);
		offset = width - 7;
	}
	return (offset);
}

/*
 * pr_idle --
 *	Display the idle time.
 *	Returns number of excess characters that were used for long idle time.
 */
int
pr_idle(time_t idle)
{
	/* If idle more than 36 hours, print as a number of days. */
	if (idle >= 36 * 3600) {
		int days = idle / 86400;
		xo_emit(" {:idle/%dday%s} ", days, days > 1 ? "s" : " " );
		if (days >= 100)
			return (2);
		if (days >= 10)
			return (1);
	}

	/* If idle more than an hour, print as HH:MM. */
	else if (idle >= 3600)
		xo_emit(" {:idle/%2d:%02d/} ",
		    (int)(idle / 3600), (int)((idle % 3600) / 60));

	else if (idle / 60 == 0)
		xo_emit("     - ");

	/* Else print the minutes idle. */
	else
		xo_emit("    {:idle/%2d} ", (int)(idle / 60));

	return (0); /* not idle longer than 9 days */
}
