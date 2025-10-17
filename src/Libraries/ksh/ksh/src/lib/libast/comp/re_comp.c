/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
 * re_comp implementation
 */

#include <ast.h>
#include <re_comp.h>
#include <regex.h>

#undef	error
#undef	valid

static struct
{
	char	error[64];
	regex_t	re;
	int	valid;
} state;

char*
re_comp(const char* pattern)
{
	register int	r;

	if (!pattern || !*pattern)
	{
		if (state.valid)
			return 0;
		r = REG_BADPAT;
	}
	else
	{
		if (state.valid)
		{
			state.valid = 0;
			regfree(&state.re);
		}
		if (!(r = regcomp(&state.re, pattern, REG_LENIENT|REG_NOSUB|REG_NULL)))
		{
			state.valid = 1;
			return 0;
		}
	}
	regerror(r, &state.re, state.error, sizeof(state.error));
	return state.error;
}

int
re_exec(const char* subject)
{
	if (state.valid && subject)
		switch (regexec(&state.re, subject, 0, NiL, 0))
		{
		case 0:
			return 1;
		case REG_NOMATCH:
			return 0;
		}
	return -1;
}
