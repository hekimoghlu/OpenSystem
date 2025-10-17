/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
 * mkfifo
 */

static const char usage[] =
"[-?\n@(#)$Id: mkfifo (AT&T Research) 2009-01-02 $\n]"
USAGE_LICENSE
"[+NAME?mkfifo - make FIFOs (named pipes)]"
"[+DESCRIPTION?\bmkfifo\b creates one or more FIFO's.  By "
	"default, the mode of created FIFO is \ba=rw\b minus the "
	"bits set in the \bumask\b(1).]"
"[m:mode]:[mode?Set the mode of created FIFO to \amode\a.  "
	"\amode\a is symbolic or octal mode as in \bchmod\b(1).  Relative "
	"modes assume an initial mode of \ba=rw\b.]"
"\n"
"\nfile ...\n"
"\n"
"[+EXIT STATUS?]{"
        "[+0?All FIFO's created successfully.]"
        "[+>0?One or more FIFO's could not be created.]"
"}"
"[+SEE ALSO?\bchmod\b(1), \bumask\b(1)]"
;

#include <cmd.h>
#include <ls.h>

int
b_mkfifo(int argc, char** argv, Shbltin_t* context)
{
	register char*	arg;
	register mode_t mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH;
	register mode_t	mask = 0;
	register int	mflag = 0;

	cmdinit(argc, argv, context, ERROR_CATALOG, 0);
	for (;;)
	{
		switch (optget(argv, usage))
		{
		case 'm':
			mflag = 1;
			mode = strperm(arg = opt_info.arg, &opt_info.arg, mode);
			if (*opt_info.arg)
				error(ERROR_exit(0), "%s: invalid mode", arg);
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
	argv += opt_info.index;
	if (error_info.errors || !*argv)
		error(ERROR_usage(2), "%s", optusage(NiL));
	mask = umask(0);
	if (!mflag)
	{
		mode &= ~mask;
		umask(mask);
		mask = 0;
	}
	while (arg = *argv++)
		if (mkfifo(arg, mode) < 0)
			error(ERROR_system(0), "%s:", arg);
	if (mask)
		umask(mask);
	return error_info.errors != 0;
}
