/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
static char sccsid[] = "@(#)strerror.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/string/strsignal.c,v 1.9 2010/01/24 10:35:26 ume Exp $");

#include "namespace.h"
#if defined(NLS)
#include <nl_types.h>
#endif
#include <limits.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include "reentrant.h"
#include "un-namespace.h"

#define	UPREFIX		"Unknown signal"

/*
 * Define a buffer size big enough to describe a 64-bit signed integer
 * converted to ASCII decimal (19 bytes), with an optional leading sign
 * (1 byte); delimiter (": ", 2 bytes); and a trailing NUL (1 byte).
 */
#define TMPSIZE	(19 + 1 + 2 + 1)
#define EBUFSIZE NL_TEXTMAX * sizeof(char)

static once_t		sig_init_once = ONCE_INITIALIZER;
static thread_key_t	sig_key;
static int		sig_keycreated = 0;

static void
sig_keycreate(void)
{
	sig_keycreated = (thr_keycreate(&sig_key, free) == 0);
}

static char *
sig_tlsalloc(void)
{
	char *ebuf = NULL;

	if (thr_once(&sig_init_once, sig_keycreate) != 0 ||
	    !sig_keycreated)
		goto thr_err;
	if ((ebuf = thr_getspecific(sig_key)) == NULL) {
		if ((ebuf = malloc(EBUFSIZE)) == NULL)
			goto thr_err;
		if (thr_setspecific(sig_key, ebuf) != 0) {
			free(ebuf);
			ebuf = NULL;
			goto thr_err;
		}
	}
thr_err:
	return (ebuf);
}

int
strsignal_r(int num, char *strsignalbuf, size_t buflen)
{
	int retval = 0;
	char tmp[TMPSIZE] = { 0 };
	size_t n;
	int signum;
	char *t, *p;

	signum = num;
	if (num < 0) {
		signum = -signum;
	}

	t = tmp;
	do {
		*t++ = "0123456789"[signum % 10];
	} while (signum /= 10);
	if (num < 0) {
		*t++ = '-';
	}
	int suffixlen = strlen(tmp) + 2;

	if (num > 0 && num < NSIG) {
		n = strlcpy(strsignalbuf,
			sys_siglist[num],
			buflen);
		if (n >= (buflen - suffixlen)) {
			retval = ERANGE;
		}
	} else {
		n = strlcpy(strsignalbuf,
			UPREFIX,
			buflen);
		retval = EINVAL;
	}

	if (n < (buflen - suffixlen)) {
		p = (strsignalbuf + n);
		*p++ = ':';
		*p++ = ' ';

		for (;;) {
			*p++ = *--t;
			if (t <= tmp)
				break;
		}
		*p = '\0';
	}

	return retval;
}

/* XXX: negative 'num' ? (REGR) */
char *
strsignal(int num)
{
	char *ebuf;

	ebuf = sig_tlsalloc();
	if (ebuf == NULL) {
		errno = ENOMEM;
		return NULL;
	}

	if (strsignal_r(num, ebuf, EBUFSIZE)) {
		errno = EINVAL;
	}

	return (ebuf);
}
