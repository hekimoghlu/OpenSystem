/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
 * AT&T Research
 *
 * logname
 */

static const char usage[] =
"[-?\n@(#)$Id: logname (AT&T Research) 1999-04-30 $\n]"
USAGE_LICENSE
"[+NAME?logname - return the user's login name]"
"[+DESCRIPTION?\blogname\b writes the users's login name to standard "
	"output.  The login name is the string that is returned by the "
	"\bgetlogin\b(2) function.  If \bgetlogin\b(2) does not return "
	"successfully, the corresponding to the real user id of the calling "
	"process is used instead.]"

"\n"
"\n\n"
"\n"
"[+EXIT STATUS?]{"
        "[+0?Successful Completion.]"
        "[+>0?An error occurred.]"
"}"
"[+SEE ALSO?\bgetlogin\b(2)]"
;


#include <cmd.h>

int
b_logname(int argc, char** argv, Shbltin_t* context)
{
	register char*	logname;

	cmdinit(argc, argv, context, ERROR_CATALOG, 0);
	for (;;)
	{
		switch (optget(argv, usage))
		{
		case ':':
			error(2, "%s", opt_info.arg);
			continue;
		case '?':
			error(ERROR_usage(2), "%s", opt_info.arg);
			continue;
		}
		break;
	}
	if (error_info.errors)
		error(ERROR_usage(2), "%s", optusage(NiL));
	if (!(logname = getlogin()))
		logname = fmtuid(getuid());
	sfputr(sfstdout, logname, '\n');
	return 0;
}

