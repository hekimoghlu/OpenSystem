/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
 * @OSF_COPYRIGHT@
 */
/*
 *	File:		clock_types.h
 *	Purpose:	Clock facility header definitions. These
 *				definitons are needed by both kernel and
 *				user-level software.
 */

/*
 *	All interfaces defined here are obsolete.
 */

#ifndef _MACH_CLOCK_TYPES_H_
#define _MACH_CLOCK_TYPES_H_

#include <stdint.h>
#include <mach/time_value.h>

/*
 * Type definitions.
 */
typedef int     alarm_type_t;           /* alarm time type */
typedef int     sleep_type_t;           /* sleep time type */
typedef int     clock_id_t;             /* clock identification type */
typedef int     clock_flavor_t;         /* clock flavor type */
typedef int     *clock_attr_t;          /* clock attribute type */
typedef int     clock_res_t;            /* clock resolution type */

/*
 * Normal time specification used by the kernel clock facility.
 */
struct mach_timespec {
	unsigned int    tv_sec;                 /* seconds */
	clock_res_t     tv_nsec;                /* nanoseconds */
};
typedef struct mach_timespec    mach_timespec_t;

/*
 * Reserved clock id values for default clocks.
 */
#define SYSTEM_CLOCK            0
#define CALENDAR_CLOCK          1

#define REALTIME_CLOCK          0

/*
 * Attribute names.
 */
#define CLOCK_GET_TIME_RES      1       /* get_time call resolution */
/*							2	 * was map_time call resolution */
#define CLOCK_ALARM_CURRES      3       /* current alarm resolution */
#define CLOCK_ALARM_MINRES      4       /* minimum alarm resolution */
#define CLOCK_ALARM_MAXRES      5       /* maximum alarm resolution */

#define NSEC_PER_USEC   1000ull         /* nanoseconds per microsecond */
#define USEC_PER_SEC    1000000ull      /* microseconds per second */
#define NSEC_PER_SEC    1000000000ull   /* nanoseconds per second */
#define NSEC_PER_MSEC   1000000ull      /* nanoseconds per millisecond */

#define BAD_MACH_TIMESPEC(t)                                       \
	((t)->tv_nsec < 0 || (t)->tv_nsec >= (long)NSEC_PER_SEC)

/* t1 <=> t2, also (t1 - t2) in nsec with max of +- 1 sec */
#define CMP_MACH_TIMESPEC(t1, t2)                                  \
	((t1)->tv_sec > (t2)->tv_sec ? (long) +NSEC_PER_SEC :          \
	((t1)->tv_sec < (t2)->tv_sec ? (long) -NSEC_PER_SEC :          \
	                (t1)->tv_nsec - (t2)->tv_nsec))

/* t1  += t2 */
#define ADD_MACH_TIMESPEC(t1, t2)                                  \
  do {                                                             \
	if (((t1)->tv_nsec += (t2)->tv_nsec) >= (long) NSEC_PER_SEC) { \
	        (t1)->tv_nsec -= (long) NSEC_PER_SEC;                  \
	        (t1)->tv_sec  += 1;                                    \
	}                                                              \
	(t1)->tv_sec += (t2)->tv_sec;                                  \
  } while (0)

/* t1  -= t2 */
#define SUB_MACH_TIMESPEC(t1, t2)                                  \
  do {                                                             \
	if (((t1)->tv_nsec -= (t2)->tv_nsec) < 0) {                    \
	        (t1)->tv_nsec += (long) NSEC_PER_SEC;                  \
	        (t1)->tv_sec  -= 1;                                    \
	}                                                              \
	(t1)->tv_sec -= (t2)->tv_sec;                                  \
  } while (0)

/*
 * Alarm parameter defines.
 */
#define ALRMTYPE                0xff            /* type (8-bit field) */
#define TIME_ABSOLUTE           0x00            /* absolute time */
#define TIME_RELATIVE           0x01            /* relative time */

#define BAD_ALRMTYPE(t)         (((t) &~ TIME_RELATIVE) != 0)

#endif /* _MACH_CLOCK_TYPES_H_ */
