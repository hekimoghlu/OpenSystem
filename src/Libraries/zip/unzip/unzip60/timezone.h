/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#ifndef __timezone_h
#define __timezone_h

#ifndef IZ_MKTIME_ONLY

/* limits for our timezone info data:
 * we support only basic standard and daylight time, with max 2 transitions
 * per year, but for the maximum range of years a 32-bit second counter
 * can cover (these are 136 years plus a bit more than one month)
 */
#define TZ_MAX_TIMES    272 /* (=2*(LastGoodYr + 1 - FirstGoodYr) */
#define TZ_MAX_TYPES    2   /* We only support basic standard and daylight */
#ifdef WIN32    /* Win32 tzinfo supplies at max (2 * 32) chars of tz names */
#define TZ_MAX_CHARS    64  /* Maximum number of abbreviation characters */
#else
#define TZ_MAX_CHARS    50  /* Maximum number of abbreviation characters */
#endif

/* supported types of transition rules */
#define JULIAN_DAY              0   /* Jn - Julian day */
#define DAY_OF_YEAR             1   /* n - day of year */
#define MONTH_NTH_DAY_OF_WEEK   2   /* Mm.n.d - month, week, day of week */


struct ttinfo {
    long            tt_gmtoff;  /* UTC offset in seconds */
    int             tt_isdst;   /* used to set tm_isdst */
    int             tt_abbrind; /* abbreviation list index */
};

struct state {
    int             timecnt;
    int             typecnt;
    int             charcnt;
    time_t          ats[TZ_MAX_TIMES];
    unsigned char   types[TZ_MAX_TIMES];
    struct ttinfo   ttis[TZ_MAX_TYPES];
    char            chars[TZ_MAX_CHARS];
};

struct rule {
    int             r_type;     /* type of rule--JULIAN_DAY etc */
    int             r_day;      /* day number of rule */
    int             r_week;     /* week number of rule */
    int             r_mon;      /* month number of rule */
    long            r_time;     /* transition time of rule */
};

extern int real_timezone_is_set;        /* set by tzset() */


/* prototypes of functions not in time.h */

void __tzset OF((void));

#ifdef NEED__ISINDST
int _isindst OF((struct tm *tb));
#endif

/* callback function to be supplied by the program that uses this library */
int GetPlatformLocalTimezone OF((register struct state * ZCONST sp,
        void (*fill_tzstate_from_rules)(struct state * ZCONST sp_res,
                                        ZCONST struct rule * ZCONST start,
                                        ZCONST struct rule * ZCONST end)));
#ifdef IZTZ_SETLOCALTZINFO
void set_TZ OF((long time_zone, int day_light));
#endif

#endif /* !IZ_MKTIME_ONLY */

time_t mkgmtime OF((struct tm *tm));

#endif
