/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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
#ifndef __APPLE__ 
#ifndef lint
static const char rcsid[] = "$Id: ns_samedomain.c,v 1.6 2005/04/27 04:56:40 sra Exp $";
#endif
#endif /* __APPLE__ */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "port_before.h"

#include <sys/types.h>
#include <arpa/nameser.h>
#include <errno.h>
#include <string.h>

#include "port_after.h"

/*%
 *	Check whether a name belongs to a domain.
 *
 * Inputs:
 *\li	a - the domain whose ancestry is being verified
 *\li	b - the potential ancestor we're checking against
 *
 * Return:
 *\li	boolean - is a at or below b?
 *
 * Notes:
 *\li	Trailing dots are first removed from name and domain.
 *	Always compare complete subdomains, not only whether the
 *	domain name is the trailing string of the given name.
 *
 *\li	"host.foobar.top" lies in "foobar.top" and in "top" and in ""
 *	but NOT in "bar.top"
 */

int
ns_samedomain(const char *a, const char *b) {
	size_t la, lb;
	size_t diff, i;
	int escaped;
	const char *cp;

	la = strlen(a);
	lb = strlen(b);

	/* Ignore a trailing label separator (i.e. an unescaped dot) in 'a'. */
	if (la != 0U && a[la - 1] == '.') {
		escaped = 0;
		/* Note this loop doesn't get executed if la==1. */
		for (i = la - 2; i >= 0; i--)
			if (a[i] == '\\') {
				if (escaped)
					escaped = 0;
				else
					escaped = 1;
			} else
				break;
		if (!escaped)
			la--;
	}

	/* Ignore a trailing label separator (i.e. an unescaped dot) in 'b'. */
	if (lb != 0U && b[lb - 1] == '.') {
		escaped = 0;
		/* note this loop doesn't get executed if lb==1 */
		for (i = lb - 2; i >= 0; i--)
			if (b[i] == '\\') {
				if (escaped)
					escaped = 0;
				else
					escaped = 1;
			} else
				break;
		if (!escaped)
			lb--;
	}

	/* lb == 0 means 'b' is the root domain, so 'a' must be in 'b'. */
	if (lb == 0U)
		return (1);

	/* 'b' longer than 'a' means 'a' can't be in 'b'. */
	if (lb > la)
		return (0);

	/* 'a' and 'b' being equal at this point indicates sameness. */
	if (lb == la)
		return (strncasecmp(a, b, lb) == 0);

	/* Ok, we know la > lb. */

	diff = la - lb;

	/*
	 * If 'a' is only 1 character longer than 'b', then it can't be
	 * a subdomain of 'b' (because of the need for the '.' label
	 * separator).
	 */
	if (diff < 2)
		return (0);

	/*
	 * If the character before the last 'lb' characters of 'b'
	 * isn't '.', then it can't be a match (this lets us avoid
	 * having "foobar.com" match "bar.com").
	 */
	if (a[diff - 1] != '.')
		return (0);

	/*
	 * We're not sure about that '.', however.  It could be escaped
         * and thus not a really a label separator.
	 */
	escaped = 0;
	for (i = diff - 2; i >= 0; i--)
		if (a[i] == '\\') {
			if (escaped)
				escaped = 0;
			else
				escaped = 1;
		} else
			break;
	if (escaped)
		return (0);
	  
	/* Now compare aligned trailing substring. */
	cp = a + diff;
	return (strncasecmp(cp, b, lb) == 0);
}

#ifndef _LIBC
/*%
 *	is "a" a subdomain of "b"?
 */
int
ns_subdomain(const char *a, const char *b) {
	return (ns_samename(a, b) != 1 && ns_samedomain(a, b));
}
#endif

/*%
 *	make a canonical copy of domain name "src"
 *
 * notes:
 * \code
 *	foo -> foo.
 *	foo. -> foo.
 *	foo.. -> foo.
 *	foo\. -> foo\..
 *	foo\\. -> foo\\.
 * \endcode
 */

int
ns_makecanon(const char *src, char *dst, size_t dstsize) {
	size_t n = strlen(src);

	if (n + sizeof "." > dstsize) {			/*%< Note: sizeof == 2 */
		errno = EMSGSIZE;
		return (-1);
	}
	strlcpy(dst, src, dstsize);
	while (n >= 1U && dst[n - 1] == '.')		/*%< Ends in "." */
		if (n >= 2U && dst[n - 2] == '\\' &&	/*%< Ends in "\." */
		    (n < 3U || dst[n - 3] != '\\'))	/*%< But not "\\." */
			break;
		else
			dst[--n] = '\0';
	dst[n++] = '.';
	dst[n] = '\0';
	return (0);
}

/*%
 *	determine whether domain name "a" is the same as domain name "b"
 *
 * return:
 *\li	-1 on error
 *\li	0 if names differ
 *\li	1 if names are the same
 */

int
ns_samename(const char *a, const char *b) {
	char ta[NS_MAXDNAME], tb[NS_MAXDNAME];

	if (ns_makecanon(a, ta, sizeof ta) < 0 ||
	    ns_makecanon(b, tb, sizeof tb) < 0)
		return (-1);
	if (strcasecmp(ta, tb) == 0)
		return (1);
	else
		return (0);
}

/*! \file */
