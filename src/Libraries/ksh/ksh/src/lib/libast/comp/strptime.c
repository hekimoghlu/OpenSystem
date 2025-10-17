/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
 * strptime implementation
 */

#define strptime	______strptime

#include <ast.h>
#include <tmx.h>

#undef	strptime

#undef	_def_map_ast
#include <ast_map.h>

#if _lib_strptime

NoN(strptime)

#else

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern char*
strptime(const char* s, const char* format, struct tm* ts)
{
	char*	e;
	char*	f;
	time_t	t;
	Tm_t	tm;

	memset(&tm, 0, sizeof(tm));
	tm.tm_sec = ts->tm_sec;
	tm.tm_min = ts->tm_min;
	tm.tm_hour = ts->tm_hour;
	tm.tm_mday = ts->tm_mday;
	tm.tm_mon = ts->tm_mon;
	tm.tm_year = ts->tm_year;
	tm.tm_wday = ts->tm_wday;
	tm.tm_yday = ts->tm_yday;
	tm.tm_isdst = ts->tm_isdst;
	t = tmtime(&tm, TM_LOCALZONE);
	t = tmscan(s, &e, format, &f, &t, 0);
	if (e == (char*)s || *f)
		return 0;
	tmxtm(&tm, tmxclock(&t), NiL);
	ts->tm_sec = tm.tm_sec;
	ts->tm_min = tm.tm_min;
	ts->tm_hour = tm.tm_hour;
	ts->tm_mday = tm.tm_mday;
	ts->tm_mon = tm.tm_mon;
	ts->tm_year = tm.tm_year;
	ts->tm_wday = tm.tm_wday;
	ts->tm_yday = tm.tm_yday;
	ts->tm_isdst = tm.tm_isdst;
	return e;
}

#endif
