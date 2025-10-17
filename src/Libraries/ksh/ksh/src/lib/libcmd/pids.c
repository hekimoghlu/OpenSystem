/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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

#define FORMAT		"PID=%(pid)d PPID=%(ppid)d PGID=%(pgid)d TID=%(tid)d SID=%(sid)d"

static const char usage[] =
"[-?\n@(#)$Id: pids (AT&T Research) 2011-08-27 $\n]"
USAGE_LICENSE
"[+NAME?pids - list calling shell process ids]"
"[+DESCRIPTION?When invoked as a shell builtin, \bpids\b lists one or "
    "more of the calling process ids determined by \bgetpid\b(2), "
    "\bgetppid\b(2), \bgetpgrp\b(2), \btcgetpgrp\b(2) and \bgetsid\b(2). "
    "Unknown or invalid ids have the value \b-1\b.]"
"[f:format?List the ids specified by \aformat\a. \aformat\a follows "
    "\bprintf\b(3) conventions, except that \bsfio\b(3) inline ids are used "
    "instead of arguments: "
    "%[-+]][\awidth\a[.\aprecis\a[.\abase\a]]]]]](\aid\a)\achar\a. The "
    "supported \aid\as are:]:[format:=" FORMAT "]"
    "{"
        "[+pid?The process id.]"
        "[+pgid?The process group id.]"
        "[+ppid?The parent process id.]"
        "[+tid|tty?The controlling terminal id.]"
        "[+sid?The session id.]"
    "}"
"[+SEE ALSO?\bgetpid\b(2), \bgetppid\b(2), \bgetpgrp\b(2), "
    "\btcgetpgrp\b(2), \bgetsid\b(2)]"
;

#include <cmd.h>
#include <ast_tty.h>
#include <sfdisc.h>

/*
 * sfkeyprintf() lookup
 * handle==0 for heading
 */

static int
key(void* handle, Sffmt_t* fp, const char* arg, char** ps, Sflong_t* pn)
{
	register char*	s;
	int		fd;
	long		tid;

	if (!(s = fp->t_str) || streq(s, "pid"))
		*pn = getpid();
	else if (streq(s, "pgid"))
		*pn = getpgrp();
	else if (streq(s, "ppid"))
		*pn = getppid();
	else if (streq(s, "tid") || streq(s, "tty"))
	{
		for (fd = 0; fd < 3; fd++)
			if ((tid = tcgetpgrp(fd)) >= 0)
				break;
		*pn = tid;
	}
	else if (streq(s, "sid"))
#if _lib_getsid
		*pn = getsid(0);
#else
		*pn = -1;
#endif
	else if (streq(s, "format"))
		*ps = (char*)handle;
	else
	{
		error(2, "%s: unknown format identifier", s);
		return 0;
	}
	return 1;
}

int
b_pids(int argc, char** argv, Shbltin_t* context)
{
	char*			format = 0;

	cmdinit(argc, argv, context, ERROR_CATALOG, 0);
	for (;;)
	{
		switch (optget(argv, usage))
		{
		case 'f':
			format = opt_info.arg;
			continue;
		case '?':
			error(ERROR_USAGE|4, "%s", opt_info.arg);
			break;
		case ':':
			error(2, "%s", opt_info.arg);
			break;
		}
		break;
	}
	argv += opt_info.index;
	if (error_info.errors || *argv)
		error(ERROR_USAGE|4, "%s", optusage(NiL));
	if (!format)
		format = FORMAT;
	sfkeyprintf(sfstdout, format, format, key, NiL);
	sfprintf(sfstdout, "\n");
	return 0;
}
