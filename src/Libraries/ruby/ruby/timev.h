/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#ifndef RUBY_TIMEV_H
#define RUBY_TIMEV_H

PACKED_STRUCT_UNALIGNED(struct vtm {
    VALUE year; /* 2000 for example.  Integer. */
    VALUE subsecx; /* 0 <= subsecx < TIME_SCALE.  possibly Rational. */
    VALUE utc_offset; /* -3600 as -01:00 for example.  possibly Rational. */
    VALUE zone; /* "JST", "EST", "EDT", etc. as String */
    unsigned int yday:9; /* 1..366 */
    unsigned int mon:4; /* 1..12 */
    unsigned int mday:5; /* 1..31 */
    unsigned int hour:5; /* 0..23 */
    unsigned int min:6; /* 0..59 */
    unsigned int sec:6; /* 0..60 */
    unsigned int wday:3; /* 0:Sunday, 1:Monday, ..., 6:Saturday 7:init */
    unsigned int isdst:2; /* 0:StandardTime 1:DayLightSavingTime 3:init */
});

#define TIME_SCALE 1000000000

#ifndef TYPEOF_TIMEVAL_TV_SEC
# define TYPEOF_TIMEVAL_TV_SEC time_t
#endif
#ifndef TYPEOF_TIMEVAL_TV_USEC
# if INT_MAX >= 1000000
# define TYPEOF_TIMEVAL_TV_USEC int
# else
# define TYPEOF_TIMEVAL_TV_USEC long
# endif
#endif

#if SIZEOF_TIME_T == SIZEOF_LONG
typedef unsigned long unsigned_time_t;
#elif SIZEOF_TIME_T == SIZEOF_INT
typedef unsigned int unsigned_time_t;
#elif SIZEOF_TIME_T == SIZEOF_LONG_LONG
typedef unsigned LONG_LONG unsigned_time_t;
#else
# error cannot find integer type which size is same as time_t.
#endif

/* strftime.c */
#ifdef RUBY_ENCODING_H
VALUE rb_strftime_timespec(const char *format, size_t format_len, rb_encoding *enc,
                           VALUE time, const struct vtm *vtm, struct timespec *ts, int gmt);
VALUE rb_strftime(const char *format, size_t format_len, rb_encoding *enc,
                  VALUE time, const struct vtm *vtm, VALUE timev, int gmt);
#endif

/* time.c */
VALUE rb_time_zone_abbreviation(VALUE zone, VALUE time);

#endif
