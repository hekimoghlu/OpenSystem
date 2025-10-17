/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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

#include <ast.h>
#include <ast_getopt.h>

#undef	_BLD_ast	/* enable ast imports since we're user static */

#include <error.h>
#include <option.h>
#include <getopt.h>
#include <ctype.h>

static const char*		lastoptstring;
static const struct option*	lastlongopts;
static char*			usage;
static Sfio_t*			up;

static int			lastoptind;

static int
golly(int argc, char* const* argv, const char* optstring, const struct option* longopts, int* longindex, int flags)
{
	register char*			s;
	register const struct option*	o;
	register int			c;
	char*				t;

	if (!up || optstring != lastoptstring || longopts != lastlongopts)
	{
		if (!up && !(up = sfstropen()) || !(t = strdup(optstring)))
			return -1;
		sfprintf(up, "[-1p%d]", flags);
		for (o = longopts; o->name; o++)
		{
			if (o->flag || o->val <= 0 || o->val > UCHAR_MAX || !isalnum(o->val))
				sfprintf(up, "\n[%d:%s]", UCHAR_MAX + 1 + (o - longopts), o->name);
			else
			{
				sfprintf(up, "\n[%c:%s]", o->val, o->name);
				if (s = strchr(t, o->val))
				{
					*s++ = ' ';
					if (*s == ':')
					{
						*s++ = ' ';
						if (*s == ':')
							*s = ' ';
					}
				}
			}
			if (o->has_arg)
			{
				sfputc(up, ':');
				if (o->has_arg == optional_argument)
					sfputc(up, '?');
				sfprintf(up, "[string]");
			}
		}
		s = t;
		while (c = *s++)
			if (c != ' ')
			{
				sfprintf(up, "\n[%c]", c);
				if (*s == ':')
				{
					sfputc(up, *s);
					if (*++s == ':')
					{
						sfputc(up, '?');
						s++;
					}
					sfputc(up, '[');
					sfputc(up, ']');
				}
			}
		sfputc(up, '\n');
		free(t);
		if (!(usage = sfstruse(up)))
			return -1;
		lastoptstring = optstring;
		lastlongopts = longopts;
	}
	opt_info.index = (optind > 1 || optind == lastoptind) ? optind : 0;
	if (opt_info.index >= argc || !(c = optget((char**)argv, usage)))
	{
		sfstrclose(up);
		up = 0;
		c = -1;
	}
	else
	{
		if (c == ':' || c == '?')
		{
			if (opterr && (!optstring || *optstring != ':'))
			{
				if (!error_info.id)
					error_info.id = argv[0];
				errormsg(NiL, c == '?' ? (ERROR_USAGE|4) : 2, "%s", opt_info.arg);
			}
			optopt = opt_info.option[1];
			c = '?';
		}
		optarg = opt_info.arg;
		if (c < 0)
		{
			o = longopts - c - UCHAR_MAX - 1;
			if (o->flag)
			{
				*o->flag = o->val;
				c = 0;
			}
			else
				c = o->val;
		}
	}
	lastoptind = optind = opt_info.index;
	return c;
}

extern int
getopt_long(int argc, char* const* argv, const char* optstring, const struct option* longopts, int* longindex)
{
	return golly(argc, argv, optstring, longopts, longindex, 2);
}

extern int
getopt_long_only(int argc, char* const* argv, const char* optstring, const struct option* longopts, int* longindex)
{
	return golly(argc, argv, optstring, longopts, longindex, 1);
}
