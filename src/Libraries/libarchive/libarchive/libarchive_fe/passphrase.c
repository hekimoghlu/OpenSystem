/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
 *//*
 * Copyright (c) 2000-2002, 2007, 2010
 * 	Todd C. Miller <millert@openbsd.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * Sponsored in part by the Defense Advanced Research Projects
 * Agency (DARPA) and Air Force Research Laboratory, Air Force
 * Materiel Command, USAF, under agreement number F39502-99-1-0512.
 */
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
 */#include "lafe_platform.h"
#include <errno.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_READPASSPHRASE_H
#include <readpassphrase.h>
#endif

#include "err.h"
#include "passphrase.h"

#ifndef HAVE_READPASSPHRASE

#define RPP_ECHO_OFF    0x00		/* Turn off echo (default). */
#define RPP_ECHO_ON     0x01		/* Leave echo on. */
#define RPP_REQUIRE_TTY 0x02		/* Fail if there is no tty. */
#define RPP_FORCELOWER  0x04		/* Force input to lower case. */
#define RPP_FORCEUPPER  0x08		/* Force input to upper case. */
#define RPP_SEVENBIT    0x10		/* Strip the high bit from input. */
#define RPP_STDIN       0x20		/* Read from stdin, not /dev/tty */


#if defined(_WIN32) && !defined(__CYGWIN__)
#include <string.h>
#include <windows.h>

static char *
readpassphrase(const char *prompt, char *buf, size_t bufsiz, int flags)
{
	HANDLE hStdin, hStdout;
	DWORD mode, rbytes;
	BOOL success;

	(void)flags;

	hStdin = GetStdHandle(STD_INPUT_HANDLE);
	if (hStdin == INVALID_HANDLE_VALUE)
		return (NULL);
	hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hStdout == INVALID_HANDLE_VALUE)
		return (NULL);

	success = GetConsoleMode(hStdin, &mode);
	if (!success)
		return (NULL);
	mode &= ~ENABLE_ECHO_INPUT;
	mode |= ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT;
	success = SetConsoleMode(hStdin, mode);
	if (!success)
		return (NULL);

	success = WriteFile(hStdout, prompt, (DWORD)strlen(prompt),
		NULL, NULL);
	if (!success)
		return (NULL);
	success = ReadFile(hStdin, buf, (DWORD)bufsiz - 1, &rbytes, NULL);
	if (!success)
		return (NULL);
	WriteFile(hStdout, "\r\n", 2, NULL, NULL);
	buf[rbytes] = '\0';
	/* Remove trailing carriage return(s). */
	buf[strcspn(buf, "\r\n")] = '\0';

	return (buf);
}

#else /* _WIN32 && !__CYGWIN__ */

#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#ifdef HAVE_PATHS_H
#include <paths.h>
#endif
#include <signal.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#ifndef _PATH_TTY
#define _PATH_TTY "/dev/tty"
#endif

#ifdef TCSASOFT
# define _T_FLUSH	(TCSAFLUSH|TCSASOFT)
#else
# define _T_FLUSH	(TCSAFLUSH)
#endif

/* SunOS 4.x which lacks _POSIX_VDISABLE, but has VDISABLE */
#if !defined(_POSIX_VDISABLE) && defined(VDISABLE)
#  define _POSIX_VDISABLE       VDISABLE
#endif

#define M(a,b) (a > b ? a : b)
#define MAX_SIGNO M(M(M(SIGALRM, SIGHUP), \
                      M(SIGINT, SIGPIPE)), \
                    M(M(SIGQUIT, SIGTERM), \
                      M(M(SIGTSTP, SIGTTIN), SIGTTOU)))

static volatile sig_atomic_t signo[MAX_SIGNO + 1];

static void
handler(int s)
{
	assert(s <= MAX_SIGNO);
	signo[s] = 1;
}

static char *
readpassphrase(const char *prompt, char *buf, size_t bufsiz, int flags)
{
	ssize_t nr;
	int input, output, save_errno, i, need_restart;
	char ch, *p, *end;
	struct termios term, oterm;
#ifdef HAVE_SIGACTION
	struct sigaction sa, savealrm, saveint, savehup, savequit, saveterm;
	struct sigaction savetstp, savettin, savettou, savepipe;
#endif

	/* I suppose we could alloc on demand in this case (XXX). */
	if (bufsiz == 0) {
		errno = EINVAL;
		return(NULL);
	}

restart:
	for (i = 0; i <= MAX_SIGNO; i++)
		signo[i] = 0;
	nr = -1;
	save_errno = 0;
	need_restart = 0;
	/*
	 * Read and write to /dev/tty if available.  If not, read from
	 * stdin and write to stderr unless a tty is required.
	 */
	if ((flags & RPP_STDIN) ||
	    (input = output = open(_PATH_TTY, O_RDWR)) == -1) {
		if (flags & RPP_REQUIRE_TTY) {
			errno = ENOTTY;
			return(NULL);
		}
		input = STDIN_FILENO;
		output = STDERR_FILENO;
	}

	/*
	 * Turn off echo if possible.
	 * If we are using a tty but are not the foreground pgrp this will
	 * generate SIGTTOU, so do it *before* installing the signal handlers.
	 */
	if (input != STDIN_FILENO && tcgetattr(input, &oterm) == 0) {
		memcpy(&term, &oterm, sizeof(term));
		if (!(flags & RPP_ECHO_ON))
			term.c_lflag &= ~(ECHO | ECHONL);
#ifdef VSTATUS
		if (term.c_cc[VSTATUS] != _POSIX_VDISABLE)
			term.c_cc[VSTATUS] = _POSIX_VDISABLE;
#endif
		(void)tcsetattr(input, _T_FLUSH, &term);
	} else {
		memset(&term, 0, sizeof(term));
		term.c_lflag |= ECHO;
		memset(&oterm, 0, sizeof(oterm));
		oterm.c_lflag |= ECHO;
	}

#ifdef HAVE_SIGACTION
	/*
	 * Catch signals that would otherwise cause the user to end
	 * up with echo turned off in the shell.  Don't worry about
	 * things like SIGXCPU and SIGVTALRM for now.
	 */
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;		/* don't restart system calls */
	sa.sa_handler = handler;
	/* Keep this list in sync with MAX_SIGNO! */
	(void)sigaction(SIGALRM, &sa, &savealrm);
	(void)sigaction(SIGHUP, &sa, &savehup);
	(void)sigaction(SIGINT, &sa, &saveint);
	(void)sigaction(SIGPIPE, &sa, &savepipe);
	(void)sigaction(SIGQUIT, &sa, &savequit);
	(void)sigaction(SIGTERM, &sa, &saveterm);
	(void)sigaction(SIGTSTP, &sa, &savetstp);
	(void)sigaction(SIGTTIN, &sa, &savettin);
	(void)sigaction(SIGTTOU, &sa, &savettou);
#endif

	if (!(flags & RPP_STDIN)) {
		int r = write(output, prompt, strlen(prompt));
		(void)r;
	}
	end = buf + bufsiz - 1;
	p = buf;
	while ((nr = read(input, &ch, 1)) == 1 && ch != '\n' && ch != '\r') {
		if (p < end) {
			if ((flags & RPP_SEVENBIT))
				ch &= 0x7f;
			if (isalpha((unsigned char)ch)) {
				if ((flags & RPP_FORCELOWER))
					ch = (char)tolower((unsigned char)ch);
				if ((flags & RPP_FORCEUPPER))
					ch = (char)toupper((unsigned char)ch);
			}
			*p++ = ch;
		}
	}
	*p = '\0';
	save_errno = errno;
	if (!(term.c_lflag & ECHO)) {
		int r = write(output, "\n", 1);
		(void)r;
	}

	/* Restore old terminal settings and signals. */
	if (memcmp(&term, &oterm, sizeof(term)) != 0) {
		const int sigttou = signo[SIGTTOU];

		/* Ignore SIGTTOU generated when we are not the fg pgrp. */
		while (tcsetattr(input, _T_FLUSH, &oterm) == -1 &&
		    errno == EINTR && !signo[SIGTTOU])
			continue;
		signo[SIGTTOU] = sigttou;
	}
#ifdef HAVE_SIGACTION
	(void)sigaction(SIGALRM, &savealrm, NULL);
	(void)sigaction(SIGHUP, &savehup, NULL);
	(void)sigaction(SIGINT, &saveint, NULL);
	(void)sigaction(SIGQUIT, &savequit, NULL);
	(void)sigaction(SIGPIPE, &savepipe, NULL);
	(void)sigaction(SIGTERM, &saveterm, NULL);
	(void)sigaction(SIGTSTP, &savetstp, NULL);
	(void)sigaction(SIGTTIN, &savettin, NULL);
	(void)sigaction(SIGTTOU, &savettou, NULL);
#endif
	if (input != STDIN_FILENO)
		(void)close(input);

	/*
	 * If we were interrupted by a signal, resend it to ourselves
	 * now that we have restored the signal handlers.
	 */
	for (i = 0; i <= MAX_SIGNO; i++) {
		if (signo[i]) {
			kill(getpid(), i);
			switch (i) {
			case SIGTSTP:
			case SIGTTIN:
			case SIGTTOU:
				need_restart = 1;
			}
		}
	}
	if (need_restart)
		goto restart;

	if (save_errno)
		errno = save_errno;
	return(nr == -1 ? NULL : buf);
}
#endif /* _WIN32 && !__CYGWIN__ */
#endif /* HAVE_READPASSPHRASE */

char *
lafe_readpassphrase(const char *prompt, char *buf, size_t bufsiz)
{
	char *p;

	p = readpassphrase(prompt, buf, bufsiz, RPP_ECHO_OFF);
	if (p == NULL) {
		switch (errno) {
		case EINTR:
			break;
		default:
			lafe_errc(1, errno, "Couldn't read passphrase");
			/* NOTREACHED */
		}
	}
	return (p);
}


