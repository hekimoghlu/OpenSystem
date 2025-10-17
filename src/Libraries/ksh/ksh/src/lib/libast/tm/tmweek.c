/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
#pragma prototyped
/*
 * Glenn Fowler
 * AT&T Research
 *
 * Time_t conversion support
 */

#include <tmx.h>

static unsigned char	offset[7][3] =
{
	{ 7, 6, 6 },
	{ 1, 7, 7 },
	{ 2, 1, 8 },
	{ 3, 2, 9 },
	{ 4, 3, 10},
	{ 5, 4, 4 },
	{ 6, 5, 5 },
};

/*
 * type is week type
 *	0 sunday first day of week
 *	1 monday first day of week
 *	2 monday first day of iso week
 * if week<0 then return week for tm
 * if day<0 then set tm to first day of week
 * otherwise set tm to day in week
 * and return tm->tm_yday
 */

int
tmweek(Tm_t* tm, int type, int week, int day)
{
	int	d;

	if (week < 0)
	{
		if ((day = tm->tm_wday - tm->tm_yday % 7) < 0)
			day += 7;
		week = (tm->tm_yday + offset[day][type]) / 7;
		if (type == 2)
		{
			if (!week)
				week = (day > 0 && day < 6 || tmisleapyear(tm->tm_year - 1)) ? 53 : 52;
			else if (week == 53 && (tm->tm_wday + (31 - tm->tm_mday)) < 4)
				week = 1;
		}
		return week;
	}
	if (day < 0)
		day = type != 0;
	tm->tm_mon = 0;
	tm->tm_mday = 1;
	tmfix(tm);
	d = tm->tm_wday;
	tm->tm_mday = week * 7 - offset[d][type] + ((day || type != 2) ? day : 7);
	tmfix(tm);
	if (d = tm->tm_wday - day)
	{
		tm->tm_mday -= d;
		tmfix(tm);
	}
	return tm->tm_yday;
}
