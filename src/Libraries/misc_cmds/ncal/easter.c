/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
__FBSDID("$FreeBSD$");

#include "calendar.h"

typedef struct date date;

static int	 easterodn(int y);

/* Compute Easter Sunday in Gregorian Calendar */
date *
easterg(int y, date *dt)
{
	int c, i, j, k, l, n;

	n = y % 19;
	c = y / 100;
	k = (c - 17) / 25;
	i = (c - c/4 -(c-k)/3 + 19 * n + 15) % 30;
	i = i -(i/28) * (1 - (i/28) * (29/(i + 1)) * ((21 - n)/11));
	j = (y + y/4 + i + 2 - c + c/4) % 7;
	l = i - j;
	dt->m = 3 + (l + 40) / 44;
	dt->d = l + 28 - 31*(dt->m / 4);
	dt->y = y;
	return (dt);
}

/* Compute the Gregorian date of Easter Sunday in Julian Calendar */
date *
easterog(int y, date *dt)
{

	return (gdate(easterodn(y), dt));
}

/* Compute the Julian date of Easter Sunday in Julian Calendar */
date   *
easteroj(int y, date * dt)
{

	return (jdate(easterodn(y), dt));
}

/* Compute the day number of Easter Sunday in Julian Calendar */
static int
easterodn(int y)
{
	/*
	 * Table for the easter limits in one metonic (19-year) cycle. 21
	 * to 31 is in March, 1 through 18 in April. Easter is the first
	 * sunday after the easter limit.
	 */
	int     mc[] = {5, 25, 13, 2, 22, 10, 30, 18, 7, 27, 15, 4,
		    24, 12, 1, 21, 9, 29, 17};

	/* Offset from a weekday to next sunday */
	int     ns[] = {6, 5, 4, 3, 2, 1, 7};
	date	dt;
	int     dn;

	/* Assign the easter limit of y to dt */
	dt.d = mc[y % 19];

	if (dt.d < 21)
		dt.m = 4;
	else
		dt.m = 3;

	dt.y = y;

	/* Return the next sunday after the easter limit */
	dn = ndaysj(&dt);
	return (dn + ns[weekday(dn)]);
}
