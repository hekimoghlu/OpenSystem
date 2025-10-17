/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)siglist.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/siglist.c,v 1.6 2007/01/20 08:24:01 maxim Exp $");

#include <signal.h>

const char *const sys_signame[NSIG] = {
	"Signal 0",
	"hup",				/* SIGHUP */
	"int",				/* SIGINT */
	"quit",				/* SIGQUIT */
	"ill",				/* SIGILL */
	"trap",				/* SIGTRAP */
	"abrt",				/* SIGABRT */
	"emt",				/* SIGEMT */
	"fpe",				/* SIGFPE */
	"kill",				/* SIGKILL */
	"bus",				/* SIGBUS */
	"segv",				/* SIGSEGV */
	"sys",				/* SIGSYS */
	"pipe",				/* SIGPIPE */
	"alrm",				/* SIGALRM */
	"term",				/* SIGTERM */
	"urg",				/* SIGURG */
	"stop",				/* SIGSTOP */
	"tstp",				/* SIGTSTP */
	"cont",				/* SIGCONT */
	"chld",				/* SIGCHLD */
	"ttin",				/* SIGTTIN */
	"ttou",				/* SIGTTOU */
	"io",				/* SIGIO */
	"xcpu",				/* SIGXCPU */
	"xfsz",				/* SIGXFSZ */
	"vtalrm",			/* SIGVTALRM */
	"prof",				/* SIGPROF */
	"winch",			/* SIGWINCH */
	"info",				/* SIGINFO */
	"usr1",				/* SIGUSR1 */
	"usr2"				/* SIGUSR2 */
};

const char *const sys_siglist[NSIG] = {
	"Signal 0",
	"Hangup",			/* SIGHUP */
	"Interrupt",			/* SIGINT */
	"Quit",				/* SIGQUIT */
	"Illegal instruction",		/* SIGILL */
	"Trace/BPT trap",		/* SIGTRAP */
	"Abort trap",			/* SIGABRT */
	"EMT trap",			/* SIGEMT */
	"Floating point exception",	/* SIGFPE */
	"Killed",			/* SIGKILL */
	"Bus error",			/* SIGBUS */
	"Segmentation fault",		/* SIGSEGV */
	"Bad system call",		/* SIGSYS */
	"Broken pipe",			/* SIGPIPE */
	"Alarm clock",			/* SIGALRM */
	"Terminated",			/* SIGTERM */
	"Urgent I/O condition",		/* SIGURG */
	"Suspended (signal)",		/* SIGSTOP */
	"Suspended",			/* SIGTSTP */
	"Continued",			/* SIGCONT */
	"Child exited",			/* SIGCHLD */
	"Stopped (tty input)",		/* SIGTTIN */
	"Stopped (tty output)",		/* SIGTTOU */
	"I/O possible",			/* SIGIO */
	"Cputime limit exceeded",	/* SIGXCPU */
	"Filesize limit exceeded",	/* SIGXFSZ */
	"Virtual timer expired",	/* SIGVTALRM */
	"Profiling timer expired",	/* SIGPROF */
	"Window size changes",		/* SIGWINCH */
	"Information request",		/* SIGINFO */
	"User defined signal 1",	/* SIGUSR1 */
	"User defined signal 2"		/* SIGUSR2 */
};
