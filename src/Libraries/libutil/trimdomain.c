/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include <sys/cdefs.h>

#include <sys/param.h>

#include <libutil.h>
#include <string.h>
#include <unistd.h>

void trimdomain(char *_fullhost, size_t _hostsize);

static int	isDISP(const char *);

/*-
 * Trim the current domain name from fullhost, but only if the result
 * is less than or equal to hostsize in length.
 *
 * This function understands $DISPLAY type fullhosts.
 *
 * For example:
 *
 *     trimdomain("abcde.my.domain", 5)       ->   "abcde"
 *     trimdomain("abcde.my.domain", 4)       ->   "abcde.my.domain"
 *     trimdomain("abcde.my.domain:0.0", 9)   ->   "abcde:0.0"
 *     trimdomain("abcde.my.domain:0.0", 8)   ->   "abcde.my.domain:0.0"
 */
void
trimdomain(char *fullhost, size_t hostsize)
{
	static size_t dlen;
	static int first = 1;
	static char domain[MAXHOSTNAMELEN];
	char *end, *s;
	size_t len;

	if (first) {
		/* XXX: Should we assume that our domain is this persistent ? */
		first = 0;
		if (gethostname(domain, sizeof(domain) - 1) == 0 &&
		    (s = strchr(domain, '.')) != NULL)
			memmove(domain, s + 1, strlen(s + 1) + 1);
		else
			domain[0] = '\0';
		dlen = strlen(domain);
	}

	if (domain[0] == '\0')
		return;

	s = fullhost;
	end = (char *)((uintptr_t)s + hostsize + 1);
	if ((s = memchr(s, '.', (size_t)(end - s))) != NULL) {
		if (strncasecmp(s + 1, domain, dlen) == 0) {
			if (s[dlen + 1] == '\0') {
				/* Found -- lose the domain. */
				*s = '\0';
			} else if (s[dlen + 1] == ':' &&
			    isDISP(s + dlen + 2) &&
			    (len = strlen(s + dlen + 1)) < (size_t)(end - s)) {
				/* Found -- shuffle the DISPLAY back. */
				memmove(s, s + dlen + 1, len + 1);
			}
		}
	}
}

/*
 * Is the given string NN or NN.NN where ``NN'' is an all-numeric string ?
 */
static int
isDISP(const char *disp)
{
	size_t w;
	int res;

	w = strspn(disp, "0123456789");
	res = 0;
	if (w > 0) {
		if (disp[w] == '\0')
			res = 1;	/* NN */
		else if (disp[w] == '.') {
			disp += w + 1;
			w = strspn(disp, "0123456789");
			if (w > 0 && disp[w] == '\0')
				res = 1;	/* NN.NN */
		}
	}
	return (res);
}
