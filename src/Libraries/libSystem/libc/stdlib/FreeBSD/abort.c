/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-prototypes"

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)abort.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/abort.c,v 1.11 2007/01/09 00:28:09 imp Exp $");

#include "namespace.h"
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <pthread.h>
#include <pthread_workqueue.h>
#include "un-namespace.h"

#include "libc_private.h"

#if __has_include(<CrashReporterClient.h>)
#include <CrashReporterClient.h>
#else
#define CRGetCrashLogMessage() NULL
#define CRSetCrashLogMessage(...)
#endif
#include "_simple.h"

extern void (*__cleanup)();
extern void __abort(void) __cold __dead2;

#define TIMEOUT	10000	/* 10 milliseconds */

void
abort()
{
	struct sigaction act;

	if (!CRGetCrashLogMessage())
		CRSetCrashLogMessage("abort() called");

	/*
	 * Fetch pthread_self() now, before we start masking signals.
	 * pthread_self will abort or crash if the pthread's signature
	 * appears corrupt. aborting inside abort is painful, so let's get
	 * that out of the way before we go any further.
	 */
	pthread_t self = pthread_self();

	/*
	 * POSIX requires we flush stdio buffers on abort.
	 * XXX ISO C requires that abort() be async-signal-safe.
	 */
	if (__cleanup)
		(*__cleanup)();

	sigfillset(&act.sa_mask);
	/*
	 * Don't block SIGABRT to give any handler a chance; we ignore
	 * any errors -- ISO C doesn't allow abort to return anyway.
	 */
	sigdelset(&act.sa_mask, SIGABRT);

	/*
	 * Don't block SIGSEGV since we might trigger a segfault if the pthread
	 * struct is corrupt. The end user behavior is that the program will
	 * terminate with a SIGSEGV instead of a SIGABRT which is acceptable. If
	 * the user registers a SIGSEGV handler, then they are responsible for
	 * dealing with any corruption themselves and abort may not work.
	 * rdar://48853131
	 */
	sigdelset(&act.sa_mask, SIGSEGV);
	sigdelset(&act.sa_mask, SIGBUS);

	/* <rdar://problem/7397932> abort() should call pthread_kill to deliver a signal to the aborting thread 
	 * This helps gdb focus on the thread calling abort()
	 */

	/* Block all signals on all other threads */
	sigset_t fullmask;
	sigfillset(&fullmask);
	(void)_sigprocmask(SIG_SETMASK, &fullmask, NULL);

	/* <rdar://problem/8400096> Set the workqueue killable */
	__pthread_workqueue_setkill(1);

	(void)pthread_sigmask(SIG_SETMASK, &act.sa_mask, NULL);
	(void)pthread_kill(self, SIGABRT);

	usleep(TIMEOUT); /* give time for signal to happen */

	/*
	 * If SIGABRT was ignored, or caught and the handler returns, do
	 * it again, only harder.
	 */
	 __abort();
}

__private_extern__ void
__abort()
{
	struct sigaction act;

	if (!CRGetCrashLogMessage())
		CRSetCrashLogMessage("__abort() called");
	act.sa_handler = SIG_DFL;
	act.sa_flags = 0;
	sigfillset(&act.sa_mask);
	(void)_sigaction(SIGABRT, &act, NULL);
	sigdelset(&act.sa_mask, SIGABRT);

	/* <rdar://problem/7397932> abort() should call pthread_kill to deliver a signal to the aborting thread 
	 * This helps gdb focus on the thread calling abort()
	 */

	/* Block all signals on all other threads */
	sigset_t fullmask;
	sigfillset(&fullmask);
	(void)_sigprocmask(SIG_SETMASK, &fullmask, NULL);

	/* <rdar://problem/8400096> Set the workqueue killable */
	__pthread_workqueue_setkill(1);

	(void)pthread_sigmask(SIG_SETMASK, &act.sa_mask, NULL);
	(void)pthread_kill(pthread_self(), SIGABRT);

	usleep(TIMEOUT); /* give time for signal to happen */

	/* If for some reason SIGABRT was not delivered, we exit using __builtin_trap
	 * which generates an illegal instruction on i386: <rdar://problem/8400958>
	 * and SIGTRAP on arm.
	 */
	sigfillset(&act.sa_mask);
	sigdelset(&act.sa_mask, SIGILL);
	sigdelset(&act.sa_mask, SIGTRAP);
	(void)_sigprocmask(SIG_SETMASK, &act.sa_mask, NULL);
	__builtin_trap();
}

void
abort_report_np(const char *fmt, ...)
{
	_SIMPLE_STRING s;
	va_list ap;

	if ((s = _simple_salloc()) != NULL) {
		va_start(ap, fmt);
		_simple_vsprintf(s, fmt, ap);
		va_end(ap);
		CRSetCrashLogMessage(_simple_string(s));
	} else
		CRSetCrashLogMessage(fmt); /* the format string is better than nothing */
	abort();
}
#pragma clang diagnostic pop
