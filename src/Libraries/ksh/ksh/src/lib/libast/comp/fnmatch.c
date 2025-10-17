/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
 * fnmatch implementation
 */

#include <ast_lib.h>

#include <ast.h>
#include <regex.h>
#include <fnmatch.h>

typedef struct
{
	int	fnm;		/* fnmatch flag			*/
	int	reg;		/* regex flag			*/
} Map_t;

static const Map_t	map[] =
{
	FNM_AUGMENTED,	REG_AUGMENTED,
	FNM_ICASE,	REG_ICASE,
	FNM_NOESCAPE,	REG_SHELL_ESCAPED,
	FNM_PATHNAME,	REG_SHELL_PATH,
	FNM_PERIOD,	REG_SHELL_DOT,
};

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern int
fnmatch(const char* pattern, const char* subject, register int flags)
{
	register int		reflags = REG_SHELL|REG_LEFT;
	register const Map_t*	mp;
	regex_t			re;
	regmatch_t		match;

	for (mp = map; mp < &map[elementsof(map)]; mp++)
		if (flags & mp->fnm)
			reflags |= mp->reg;
	if (flags & FNM_LEADING_DIR)
	{
		if (!(reflags = regcomp(&re, pattern, reflags)))
		{
			reflags = regexec(&re, subject, 1, &match, 0);
			regfree(&re);
			if (!reflags && (reflags = subject[match.rm_eo]))
				reflags = reflags == '/' ? 0 : FNM_NOMATCH;
		}
	}
	else if (!(reflags = regcomp(&re, pattern, reflags|REG_RIGHT)))
	{
		reflags = regexec(&re, subject, 0, NiL, 0);
		regfree(&re);
	}
	return reflags;
}
