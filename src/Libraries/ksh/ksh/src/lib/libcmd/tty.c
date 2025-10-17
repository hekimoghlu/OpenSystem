/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
 * David Korn
 * AT&T Bell Laboratories
 *
 * tty
 */

static const char usage[] =
"[-?\n@(#)$Id: tty (AT&T Research) 2008-03-13 $\n]"
USAGE_LICENSE
"[+NAME?tty - write the name of the terminal to standard output]"
"[+DESCRIPTION?\btty\b writes the name of the terminal that is connected "
	"to standard input onto standard output.  If the standard input is not "
	"a terminal, \"\bnot a tty\b\" will be written to standard output.]"
"[l:line-number?Write the synchronous line number of the terminal on a "
	"separate line following the terminal name line. If the standard "
	"input is not a synchronous  terminal then "
	"\"\bnot on an active synchronous line\b\" is written.]"
"[s:silent|quiet?Disable the terminal name line. Use \b[[ -t 0 ]]]]\b instead.]"
"[+EXIT STATUS?]{"
        "[+0?Standard input is a tty.]"
        "[+1?Standard input is not a tty.]"
        "[+2?Invalid arguments.]"
        "[+3?A an error occurred.]"
"}"
;


#include <cmd.h>

#if _mac_STWLINE
#include <sys/stermio.h>
#endif

int
b_tty(int argc, char** argv, Shbltin_t* context)
{
	register int	sflag = 0;
	register int	lflag = 0;
	register char*	tty;
#if _mac_STWLINE
	int		n;
#endif

	cmdinit(argc, argv, context, ERROR_CATALOG, 0);
	for (;;)
	{
		switch (optget(argv, usage))
		{
		case 'l':
			lflag++;
			continue;
		case 's':
			sflag++;
			continue;
		case ':':
			error(2, "%s", opt_info.arg);
			break;
		case '?':
			error(ERROR_usage(2), "%s", opt_info.arg);
			break;
		}
		break;
	}
	if(error_info.errors)
		error(ERROR_usage(2), "%s", optusage(NiL));
	if(!(tty=ttyname(0)))
	{
		tty = ERROR_translate(0, 0, 0, "not a tty");
		error_info.errors++;
	}
	if(!sflag)
		sfputr(sfstdout,tty,'\n');
	if(lflag)
	{
#if _mac_STWLINE
		if ((n = ioctl(0, STWLINE, 0)) >= 0)
			error(ERROR_OUTPUT, 1, "synchronous line %d", n);
		else
#endif
			error(ERROR_OUTPUT, 1, "not on an active synchronous line");
	}
	return(error_info.errors);
}
