/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
 * posix regex executor
 * single unsized-string interface
 */

#include "reglib.h"

/*
 * standard wrapper for the sized-record interface
 */

int
regexec(const regex_t* p, const char* s, size_t nmatch, regmatch_t* match, regflags_t flags)
{
	if (flags & REG_STARTEND)
	{
		int		r;
		int		m = match->rm_so;
		regmatch_t*	e;

		if (!(r = regnexec(p, s + m, match->rm_eo - m, nmatch, match, flags)) && m > 0)
			for (e = match + nmatch; match < e; match++)
				if (match->rm_so >= 0)
				{
					match->rm_so += m;
					match->rm_eo += m;
				}
		return r;
	}
	return regnexec(p, s, s ? strlen(s) : 0, nmatch, match, flags);
}

/*
 * 20120528: regoff_t changed from int to ssize_t
 */

#if defined(__EXPORT__)
#define extern		__EXPORT__
#endif

#undef	regexec
#if _map_libc
#define regexec		_ast_regexec
#endif

extern int
regexec(const regex_t* p, const char* s, size_t nmatch, oldregmatch_t* oldmatch, regflags_t flags)
{
	if (oldmatch)
	{
		regmatch_t*	match;
		size_t		i;
		int		r;

		if (!(match = oldof(0, regmatch_t, nmatch, 0)))
			return -1;
		if (!(r = regexec_20120528(p, s, nmatch, match, flags)))
			for (i = 0; i < nmatch; i++)
			{
				oldmatch[i].rm_so = match[i].rm_so;
				oldmatch[i].rm_eo = match[i].rm_eo;
			}
		free(match);
		return r;
	}
	return regexec_20120528(p, s, 0, NiL, flags);
}
