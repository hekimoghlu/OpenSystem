/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
 * mylog.h
 * - logging related functions
 */

#ifndef _S_MYLOG_H
#define _S_MYLOG_H

/* 
 * Modification History
 *
 * June 15, 2009		Dieter Siegmund (dieter@apple.com)
 * - split out from ipconfigd.c
 */

#if MYLOG_STDOUT
#include <SystemConfiguration/SCPrivate.h>

#define my_log(pri, format, ...)	do {		\
	struct timeval	tv;				\
	struct tm       tm;				\
	time_t		t;				\
							\
	(void)gettimeofday(&tv, NULL);					\
	t = tv.tv_sec;							\
	(void)localtime_r(&t, &tm);					\
									\
	SCPrint(TRUE, stdout,						\
		CFSTR("%04d/%02d/%02d %2d:%02d:%02d.%06d " format "\n"), \
		tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,		\
		tm.tm_hour, tm.tm_min, tm.tm_sec, tv.tv_usec,		\
		## __VA_ARGS__ );					\
    } while (0)

#define my_log_fl(pri, format, ...)	do {		\
	struct timeval	tv;				\
	struct tm       tm;				\
	time_t		t;				\
							\
	(void)gettimeofday(&tv, NULL);					\
	t = tv.tv_sec;							\
	(void)localtime_r(&t, &tm);					\
									\
	SCPrint(TRUE, stdout,						\
		CFSTR("%04d/%02d/%02d %2d:%02d:%02d.%06d " format "\n"), \
		tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,		\
		tm.tm_hour, tm.tm_min, tm.tm_sec, tv.tv_usec,		\
		## __VA_ARGS__ );					\
    } while (0)

#else /* MYLOG_STDOUT */

#include "IPConfigurationLog.h"
#define my_log		SC_log
#define my_log_fl	SC_log
#endif /* MYLOG_STDOUT */

#endif /* _S_MYLOG_H */
