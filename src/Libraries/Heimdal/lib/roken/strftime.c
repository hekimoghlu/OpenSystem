/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#include <config.h>
#include "roken.h"
#ifdef TEST_STRPFTIME
#include "strpftime-test.h"
#endif

static const char *abb_weekdays[] = {
    "Sun",
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
};

static const char *full_weekdays[] = {
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
};

static const char *abb_month[] = {
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
};

static const char *full_month[] = {
    "January",
    "February",
    "Mars",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
};

static const char *ampm[] = {
    "AM",
    "PM"
};

/*
 * Convert hour in [0, 24] to [12 1 - 11 12 1 - 11 12]
 */

static int
hour_24to12 (int hour)
{
    int ret = hour % 12;

    if (ret == 0)
	ret = 12;
    return ret;
}

/*
 * Return AM or PM for `hour'
 */

static const char *
hour_to_ampm (int hour)
{
    return ampm[hour / 12];
}

/*
 * Return the week number of `tm' (Sunday being the first day of the week)
 * as [0, 53]
 */

static int
week_number_sun (const struct tm *tm)
{
    return (tm->tm_yday + 7 - (tm->tm_yday % 7 - tm->tm_wday + 7) % 7) / 7;
}

/*
 * Return the week number of `tm' (Monday being the first day of the week)
 * as [0, 53]
 */

static int
week_number_mon (const struct tm *tm)
{
    int wday = (tm->tm_wday + 6) % 7;

    return (tm->tm_yday + 7 - (tm->tm_yday % 7 - wday + 7) % 7) / 7;
}

/*
 * Return the week number of `tm' (Monday being the first day of the
 * week) as [01, 53].  Week number one is the one that has four or more
 * days in that year.
 */

static int
week_number_mon4 (const struct tm *tm)
{
    int wday  = (tm->tm_wday + 6) % 7;
    int w1day = (wday - tm->tm_yday % 7 + 7) % 7;
    int ret;

    ret = (tm->tm_yday + w1day) / 7;
    if (w1day >= 4)
	--ret;
    if (ret == -1)
	ret = 53;
    else
	++ret;
    return ret;
}

/*
 *
 */

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
strftime (char *buf, size_t maxsize, const char *format,
	  const struct tm *tm)
{
    size_t n = 0;
    int ret;

    while (*format != '\0' && n < maxsize) {
	if (*format == '%') {
	    ++format;
	    if(*format == 'E' || *format == 'O')
		++format;
	    switch (*format) {
	    case 'a' :
		ret = snprintf (buf, maxsize - n,
				"%s", abb_weekdays[tm->tm_wday]);
		break;
	    case 'A' :
		ret = snprintf (buf, maxsize - n,
				"%s", full_weekdays[tm->tm_wday]);
		break;
	    case 'h' :
	    case 'b' :
		ret = snprintf (buf, maxsize - n,
				"%s", abb_month[tm->tm_mon]);
		break;
	    case 'B' :
		ret = snprintf (buf, maxsize - n,
				"%s", full_month[tm->tm_mon]);
		break;
	    case 'c' :
		ret = snprintf (buf, maxsize - n,
				"%d:%02d:%02d %02d:%02d:%02d",
				tm->tm_year,
				tm->tm_mon + 1,
				tm->tm_mday,
				tm->tm_hour,
				tm->tm_min,
				tm->tm_sec);
		break;
	    case 'C' :
		ret = snprintf (buf, maxsize - n,
				"%02d", (tm->tm_year + 1900) / 100);
		break;
	    case 'd' :
		ret = snprintf (buf, maxsize - n,
				"%02d", tm->tm_mday);
		break;
	    case 'D' :
		ret = snprintf (buf, maxsize - n,
				"%02d/%02d/%02d",
				tm->tm_mon + 1,
				tm->tm_mday,
				(tm->tm_year + 1900) % 100);
		break;
	    case 'e' :
		ret = snprintf (buf, maxsize - n,
				"%2d", tm->tm_mday);
		break;
	    case 'F':
		ret = snprintf (buf, maxsize - n,
				"%04d-%02d-%02d", tm->tm_year + 1900,
				tm->tm_mon + 1, tm->tm_mday);
		break;
	    case 'g':
		/* last two digits of week-based year */
		abort();
	    case 'G':
		/* week-based year */
		abort();
	    case 'H' :
		ret = snprintf (buf, maxsize - n,
				"%02d", tm->tm_hour);
		break;
	    case 'I' :
		ret = snprintf (buf, maxsize - n,
				"%02d",
				hour_24to12 (tm->tm_hour));
		break;
	    case 'j' :
		ret = snprintf (buf, maxsize - n,
				"%03d", tm->tm_yday + 1);
		break;
	    case 'k' :
		ret = snprintf (buf, maxsize - n,
				"%2d", tm->tm_hour);
		break;
	    case 'l' :
		ret = snprintf (buf, maxsize - n,
				"%2d",
				hour_24to12 (tm->tm_hour));
		break;
	    case 'm' :
		ret = snprintf (buf, maxsize - n,
				"%02d", tm->tm_mon + 1);
		break;
	    case 'M' :
		ret = snprintf (buf, maxsize - n,
				"%02d", tm->tm_min);
		break;
	    case 'n' :
		ret = snprintf (buf, maxsize - n, "\n");
		break;
	    case 'p' :
		ret = snprintf (buf, maxsize - n, "%s",
				hour_to_ampm (tm->tm_hour));
		break;
	    case 'r' :
		ret = snprintf (buf, maxsize - n,
				"%02d:%02d:%02d %s",
				hour_24to12 (tm->tm_hour),
				tm->tm_min,
				tm->tm_sec,
				hour_to_ampm (tm->tm_hour));
		break;
	    case 'R' :
		ret = snprintf (buf, maxsize - n,
				"%02d:%02d",
				tm->tm_hour,
				tm->tm_min);
		break;
	    case 's' :
		ret = snprintf (buf, maxsize - n,
				"%d", (int)mktime(rk_UNCONST(tm)));
		break;
	    case 'S' :
		ret = snprintf (buf, maxsize - n,
				"%02d", tm->tm_sec);
		break;
	    case 't' :
		ret = snprintf (buf, maxsize - n, "\t");
		break;
	    case 'T' :
	    case 'X' :
		ret = snprintf (buf, maxsize - n,
				"%02d:%02d:%02d",
				tm->tm_hour,
				tm->tm_min,
				tm->tm_sec);
		break;
	    case 'u' :
		ret = snprintf (buf, maxsize - n,
				"%d", (tm->tm_wday == 0) ? 7 : tm->tm_wday);
		break;
	    case 'U' :
		ret = snprintf (buf, maxsize - n,
				"%02d", week_number_sun (tm));
		break;
	    case 'V' :
		ret = snprintf (buf, maxsize - n,
				"%02d", week_number_mon4 (tm));
		break;
	    case 'w' :
		ret = snprintf (buf, maxsize - n,
				"%d", tm->tm_wday);
		break;
	    case 'W' :
		ret = snprintf (buf, maxsize - n,
				"%02d", week_number_mon (tm));
		break;
	    case 'x' :
		ret = snprintf (buf, maxsize - n,
				"%d:%02d:%02d",
				tm->tm_year,
				tm->tm_mon + 1,
				tm->tm_mday);
		break;
	    case 'y' :
		ret = snprintf (buf, maxsize - n,
				"%02d", (tm->tm_year + 1900) % 100);
		break;
	    case 'Y' :
		ret = snprintf (buf, maxsize - n,
				"%d", tm->tm_year + 1900);
		break;
	    case 'z':
		ret = snprintf (buf, maxsize - n,
				"%ld",
#if defined(HAVE_STRUCT_TM_TM_GMTOFF)
				(long)tm->tm_gmtoff
#elif defined(HAVE_TIMEZONE)
#ifdef HAVE_ALTZONE
				tm->tm_isdst ?
				(long)altzone :
#endif
				(long)timezone
#else
#error Where in timezone chaos are you?
#endif
				);
		break;
	    case 'Z' :
		ret = snprintf (buf, maxsize - n,
				"%s",

#if defined(HAVE_STRUCT_TM_TM_ZONE)
				tm->tm_zone
#elif defined(HAVE_TIMEZONE)
				tzname[tm->tm_isdst]
#else
#error what?
#endif
		    );
		break;
	    case '\0' :
		--format;
		/* FALLTHROUGH */
	    case '%' :
		ret = snprintf (buf, maxsize - n,
				"%%");
		break;
	    default :
		ret = snprintf (buf, maxsize - n,
				"%%%c", *format);
		break;
	    }
	    if (ret < 0 || ret >= (int)(maxsize - n))
		return 0;
	    n   += ret;
	    buf += ret;
	    ++format;
	} else {
	    *buf++ = *format++;
	    ++n;
	}
    }
    *buf = '\0';
    return n;
}
