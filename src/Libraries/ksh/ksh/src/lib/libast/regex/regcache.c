/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
 * regcomp() regex_t cache
 * at&t research
 */

#include <ast.h>
#include <regex.h>

#define CACHE		8		/* default # cached re's	*/
#define ROUND		64		/* pattern buffer size round	*/

typedef unsigned long Key_t;

typedef struct Cache_s
{
	char*		pattern;
	regex_t		re;
	unsigned long	serial;
	regflags_t	reflags;
	int		keep;
	int		size;
} Cache_t;

typedef struct State_s
{
	unsigned int	size;
	unsigned long	serial;
	char*		locale;
	Cache_t**	cache;
} State_t;

static State_t	matchstate;

/*
 * flush the cache
 */

static void
flushcache(void)
{
	register int		i;

	for (i = matchstate.size; i--;)
		if (matchstate.cache[i] && matchstate.cache[i]->keep)
		{
			matchstate.cache[i]->keep = 0;
			regfree(&matchstate.cache[i]->re);
		}
}

/*
 * return regcomp() compiled re for pattern and reflags
 */

regex_t*
regcache(const char* pattern, regflags_t reflags, int* status)
{
	register Cache_t*	cp;
	register int		i;
	char*			s;
	int			empty;
	int			unused;
	int			old;
	Key_t			key;

	/*
	 * 0 pattern flushes the cache and reflags>0 extends cache
	 */

	if (!pattern)
	{
		flushcache();
		i = 0;
		if (reflags > matchstate.size)
		{
			if (matchstate.cache = newof(matchstate.cache, Cache_t*, reflags, 0))
				matchstate.size = reflags;
			else
			{
				matchstate.size = 0;
				i = 1;
			}
		}
		if (status)
			*status = i;
		return 0;
	}
	if (!matchstate.cache)
	{
		if (!(matchstate.cache = newof(0, Cache_t*, CACHE, 0)))
			return 0;
		matchstate.size = CACHE;
	}

	/*
	 * flush the cache if the locale changed
	 * the ast setlocale() intercept maintains
	 * persistent setlocale() return values
	 */

	if ((s = setlocale(LC_CTYPE, NiL)) != matchstate.locale)
	{
		matchstate.locale = s;
		flushcache();
	}

	/*
	 * check if the pattern is in the cache
	 */

	for (i = 0; i < sizeof(key) && pattern[i]; i++)
		((char*)&key)[i] = pattern[i];
	for (; i < sizeof(key); i++)
		((char*)&key)[i] = 0;
	empty = unused = -1;
	old = 0;
	for (i = matchstate.size; i--;)
		if (!matchstate.cache[i])
			empty = i;
		else if (!matchstate.cache[i]->keep)
			unused = i;
		else if (*(Key_t*)matchstate.cache[i]->pattern == key && !strcmp(matchstate.cache[i]->pattern, pattern) && matchstate.cache[i]->reflags == reflags)
			break;
		else if (!matchstate.cache[old] || matchstate.cache[old]->serial > matchstate.cache[i]->serial)
			old = i;
	if (i < 0)
	{
		if (unused < 0)
		{
			if (empty < 0)
				unused = old;
			else
				unused = empty;
		}
		if (!(cp = matchstate.cache[unused]) && !(cp = matchstate.cache[unused] = newof(0, Cache_t, 1, 0)))
		{
			if (status)
				*status = REG_ESPACE;
			return 0;
		}
		if (cp->keep)
		{
			cp->keep = 0;
			regfree(&cp->re);
		}
		if ((i = strlen(pattern) + 1) > cp->size)
		{
			cp->size = roundof(i, ROUND);
			if (!(cp->pattern = newof(cp->pattern, char, cp->size, 0)))
			{
				if (status)
					*status = REG_ESPACE;
				return 0;
			}
		}
		strcpy(cp->pattern, pattern);
		while (++i < sizeof(Key_t))
			cp->pattern[i] = 0;
		pattern = (const char*)cp->pattern;
		if (i = regcomp(&cp->re, pattern, reflags))
		{
			if (status)
				*status = i;
			return 0;
		}
		cp->keep = 1;
		cp->reflags = reflags;
	}
	else
		cp = matchstate.cache[i];
	cp->serial = ++matchstate.serial;
	if (status)
		*status = 0;
	return &cp->re;
}
