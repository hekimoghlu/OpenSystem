/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
 * getdate implementation
 */

#define getdate	______getdate

#include <ast.h>
#include <tm.h>

#undef	getdate

#undef	_def_map_ast
#include <ast_map.h>

#undef	_lib_getdate	/* we can pass X/Open */

#if _lib_getdate

NoN(getdate)

#else

#ifndef getdate_err
__DEFINE__(int, getdate_err, 0);
#endif

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern struct tm*
getdate(const char* s)
{
	char*			e;
	char*			f;
	time_t			t;
	Tm_t*			tm;

	static struct tm	ts;

	t = tmscan(s, &e, NiL, &f, NiL, TM_PEDANTIC);
	if (*e || *f)
	{
		/* of course we all know what 7 means */
		getdate_err = 7;
		return 0;
	}
	tm = tmmake(&t);
	ts.tm_sec = tm->tm_sec;
	ts.tm_min = tm->tm_min;
	ts.tm_hour = tm->tm_hour;
	ts.tm_mday = tm->tm_mday;
	ts.tm_mon = tm->tm_mon;
	ts.tm_year = tm->tm_year;
	ts.tm_wday = tm->tm_wday;
	ts.tm_yday = tm->tm_yday;
	ts.tm_isdst = tm->tm_isdst;
	return &ts;
}

#endif
