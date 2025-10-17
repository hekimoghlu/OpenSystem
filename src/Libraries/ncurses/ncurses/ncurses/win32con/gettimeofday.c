/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#ifdef WINVER
#  undef WINVER
#endif
#define WINVER 0x0501

#include <curses.priv.h>

#include <windows.h>

MODULE_ID("$Id: gettimeofday.c,v 1.3 2014/04/26 19:41:34 juergen Exp $")

#define JAN1970 116444736000000000LL	/* the value for 01/01/1970 00:00 */

int
gettimeofday(struct timeval *tv, void *tz GCC_UNUSED)
{
    union {
	FILETIME ft;
	long long since1601;	/* time since 1 Jan 1601 in 100ns units */
    } data;

    GetSystemTimeAsFileTime(&data.ft);
    tv->tv_usec = (long) ((data.since1601 / 10LL) % 1000000LL);
    tv->tv_sec = (long) ((data.since1601 - JAN1970) / 10000000LL);
    return (0);
}
