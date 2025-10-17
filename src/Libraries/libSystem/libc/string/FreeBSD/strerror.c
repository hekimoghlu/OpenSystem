/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/string/strerror.c,v 1.16 2007/01/09 00:28:12 imp Exp $");

#if defined(NLS)
#include <nl_types.h>
#endif

#include <limits.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define	UPREFIX		"Unknown error"

/*
 * Define a buffer size big enough to describe a 64-bit signed integer
 * converted to ASCII decimal (19 bytes), with an optional leading sign
 * (1 byte); finally, we get the prefix, delimiter (": ") and a trailing
 * NUL from UPREFIX.
 */
#define	EBUFSIZE	(20 + 2 + sizeof(UPREFIX))

/*
 * Doing this by hand instead of linking with stdio(3) avoids bloat for
 * statically linked binaries.
 */
__private_extern__ void
__errstr(int num, char *uprefix, char *buf, size_t len)
{
	char *t;
	unsigned int uerr;
	char tmp[EBUFSIZE];

	t = tmp + sizeof(tmp);
	*--t = '\0';
	uerr = (num >= 0) ? num : -num;
	do {
		*--t = "0123456789"[uerr % 10];
	} while (uerr /= 10);
	if (num < 0)
		*--t = '-';
	*--t = ' ';
	*--t = ':';
	strlcpy(buf, uprefix, len);
	strlcat(buf, t, len);
}

int
strerror_r(int errnum, char *strerrbuf, size_t buflen)
{
	int retval = 0;
#if defined(NLS)
	int saved_errno = errno;
	nl_catd catd;
	catd = catopen("libc", NL_CAT_LOCALE);
#endif

	if (errnum < 0 || errnum >= sys_nerr) {
		__errstr(errnum,
#if defined(NLS)
			catgets(catd, 1, 0xffff, UPREFIX),
#else
			UPREFIX,
#endif
			strerrbuf, buflen);
		retval = EINVAL;
	} else {
		if (strlcpy(strerrbuf,
#if defined(NLS)
			catgets(catd, 1, errnum, sys_errlist[errnum]),
#else
			sys_errlist[errnum],
#endif
			buflen) >= buflen)
		retval = ERANGE;
	}

#if defined(NLS)
	catclose(catd);
	errno = saved_errno;
#endif

	return (retval);
}

static char *__strerror_ebuf = NULL;

char *
strerror(int num)
{
#if !defined(NLS)
	if (num >= 0 && num < sys_nerr) {
		return (char*)sys_errlist[num];
	}
#endif

	if (__strerror_ebuf == NULL) {
		__strerror_ebuf = calloc(1, NL_TEXTMAX);
		if (__strerror_ebuf == NULL) {
			return NULL;
		}
	}

	if (strerror_r(num, __strerror_ebuf, NL_TEXTMAX) != 0) {
		errno = EINVAL;
	}
	return __strerror_ebuf;
}
