/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
 * posix regex error message handler
 */

static const char id[] = "\n@(#)$Id: regex (AT&T Research) 2012-05-31 $\0\n";

#include "reglib.h"

static const char*	reg_error[] =
{
	/* REG_ENOSYS	*/	"not supported",
	/* REG_SUCCESS	*/	"success",
	/* REG_NOMATCH	*/	"no match",
	/* REG_BADPAT	*/	"invalid regular expression",
	/* REG_ECOLLATE	*/	"invalid collation element",
	/* REG_ECTYPE	*/	"invalid character class",
	/* REG_EESCAPE	*/	"trailing \\ in pattern",
	/* REG_ESUBREG	*/	"invalid \\digit backreference",
	/* REG_EBRACK	*/	"[...] imbalance",
	/* REG_EPAREN	*/	"\\(...\\) or (...) imbalance",
	/* REG_EBRACE	*/	"\\{...\\} or {...} imbalance",
	/* REG_BADBR	*/	"invalid {...} digits",
	/* REG_ERANGE	*/	"invalid [...] range endpoint",
	/* REG_ESPACE	*/	"out of space",
	/* REG_BADRPT	*/	"unary op not preceded by re",
	/* REG_ENULL	*/	"empty subexpr in pattern",
	/* REG_ECOUNT	*/	"re component count overflow",
	/* REG_BADESC	*/	"invalid \\char escape",
	/* REG_VERSIONID*/	&id[10],
	/* REG_EFLAGS	*/	"conflicting flags",
	/* REG_EDELIM	*/	"invalid or omitted delimiter",
	/* REG_PANIC	*/	"unrecoverable internal error",
};

size_t
regerror(int code, const regex_t* p, char* buf, size_t size)
{
	const char*	s;

	NoP(p);
	if (code++ == REG_VERSIONID)
		s = (const char*)fmtident(&id[1]);
	else if (code >= 0 && code < elementsof(reg_error))
		s = reg_error[code];
	else
		s = (const char*)"unknown error";
	if (size)
	{
		strlcpy(buf, s, size);
		buf[size - 1] = 0;
	}
	else
		buf = (char*)s;
	return strlen(buf) + 1;
}

/*
 * discipline error intercept
 */

int
fatal(regdisc_t* disc, int code, const char* pattern)
{
	if (disc->re_errorf)
	{
		if (pattern)
			(*disc->re_errorf)(NiL, disc, disc->re_errorlevel, "regular expression: %s: %s", pattern, reg_error[code+1]);
		else
			(*disc->re_errorf)(NiL, disc, disc->re_errorlevel, "regular expression: %s", reg_error[code+1]);
	}
	return code;
}
