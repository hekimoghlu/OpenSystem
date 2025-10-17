/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
static char sccsid[] = "@(#)system.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/system.c,v 1.11 2007/01/09 00:28:10 imp Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <spawn.h>
#include <paths.h>
#include <errno.h>
#include "un-namespace.h"
#include "libc_private.h"

#include <crt_externs.h>
#define environ (*_NSGetEnviron())

#include <TargetConditionals.h>

#if __DARWIN_UNIX03
#include <pthread.h>

#if !(TARGET_OS_IPHONE && (TARGET_OS_SIMULATOR || !TARGET_OS_IOS))
static pthread_mutex_t __systemfn_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif
extern int __unix_conforming;
#endif /* __DARWIN_UNIX03 */

int
__system(command)
	const char *command;
{
#if TARGET_OS_IPHONE && (TARGET_OS_SIMULATOR || !TARGET_OS_IOS)
	// Don't abort() on iOS for now
	LIBC_ABORT("system() is not supported on this platform.");
#else
	pid_t pid, savedpid;
	int pstat, err;
	struct sigaction ign, intact, quitact;
	sigset_t newsigblock, oldsigblock, defaultsig;
	posix_spawnattr_t attr;
	short flags = POSIX_SPAWN_SETSIGMASK;
	const char *argv[] = {"sh", "-c", command, NULL};

#if __DARWIN_UNIX03
	if (__unix_conforming == 0)
		__unix_conforming = 1;
#ifdef VARIANT_CANCELABLE
	pthread_testcancel();
#endif /* VARIANT_CANCELABLE */
#endif /* __DARWIN_UNIX03 */

	if (!command) {		/* just checking... */
#if TARGET_OS_IPHONE
		return(0);
#else
		if (access(_PATH_BSHELL, F_OK) == -1)	/* if no sh or no access */
			return(0);
		else
			return(1);
#endif
	}

	if ((err = posix_spawnattr_init(&attr)) != 0) {
		errno = err;
		return -1;
	}
	(void)sigemptyset(&defaultsig);

#if __DARWIN_UNIX03
	pthread_mutex_lock(&__systemfn_mutex);
#endif /* __DARWIN_UNIX03 */
	/*
	 * Ignore SIGINT and SIGQUIT, block SIGCHLD. Remember to save
	 * existing signal dispositions.
	 */
	ign.sa_handler = SIG_IGN;
	(void)sigemptyset(&ign.sa_mask);
	ign.sa_flags = 0;
	(void)_sigaction(SIGINT, &ign, &intact);
	if (intact.sa_handler != SIG_IGN) {
		sigaddset(&defaultsig, SIGINT);
		flags |= POSIX_SPAWN_SETSIGDEF;
	}
	(void)_sigaction(SIGQUIT, &ign, &quitact);
	if (quitact.sa_handler != SIG_IGN) {
		sigaddset(&defaultsig, SIGQUIT);
		flags |= POSIX_SPAWN_SETSIGDEF;
	}
	(void)sigemptyset(&newsigblock);
	(void)sigaddset(&newsigblock, SIGCHLD);
	(void)_sigprocmask(SIG_BLOCK, &newsigblock, &oldsigblock);
	(void)posix_spawnattr_setsigmask(&attr, &oldsigblock);
	if (flags & POSIX_SPAWN_SETSIGDEF) {
		(void)posix_spawnattr_setsigdefault(&attr, &defaultsig);
	}
	(void)posix_spawnattr_setflags(&attr, flags);

	err = posix_spawn(&pid, _PATH_BSHELL, NULL, &attr, (char *const *)argv, environ);
	(void)posix_spawnattr_destroy(&attr);
	if (err == 0) {
		savedpid = pid;
		do {
			pid = _wait4(savedpid, &pstat, 0, (struct rusage *)0);
		} while (pid == -1 && errno == EINTR);
		if (pid == -1) pstat = -1;
	} else if (err == ENOMEM || err == EAGAIN) { /* as if fork failed */
		pstat = -1;
	} else {
		pstat = W_EXITCODE(127, 0); /* couldn't exec shell */
	}

	(void)_sigaction(SIGINT, &intact, NULL);
	(void)_sigaction(SIGQUIT,  &quitact, NULL);
	(void)_sigprocmask(SIG_SETMASK, &oldsigblock, NULL);
#if __DARWIN_UNIX03
	pthread_mutex_unlock(&__systemfn_mutex);
#endif /* __DARWIN_UNIX03 */
	return(pstat);
#endif /* TARGET_OS_IPHONE && (TARGET_OS_SIMULATOR || !TARGET_OS_IOS) */
}

__weak_reference(__system, system);
__weak_reference(__system, _system);

#pragma clang diagnostic pop
