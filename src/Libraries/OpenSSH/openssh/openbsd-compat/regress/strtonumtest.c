/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
 * Copyright (c) 2004 Otto Moerbeek <otto@drijf.net>
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
 */

/* OPENBSD ORIGINAL: regress/lib/libc/strtonum/strtonumtest.c */

#include "includes.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

/* LLONG_MAX is known as LONGLONG_MAX on AIX */
#if defined(LONGLONG_MAX) && !defined(LLONG_MAX)
# define LLONG_MAX LONGLONG_MAX
# define LLONG_MIN LONGLONG_MIN
#endif

/* LLONG_MAX is known as LONG_LONG_MAX on HP-UX */
#if defined(LONG_LONG_MAX) && !defined(LLONG_MAX)
# define LLONG_MAX LONG_LONG_MAX
# define LLONG_MIN LONG_LONG_MIN
#endif

long long strtonum(const char *, long long, long long, const char **);

int fail;

void
test(const char *p, long long lb, long long ub, int ok)
{
	long long val;
	const char *q;

	val = strtonum(p, lb, ub, &q);
	if (ok && q != NULL) {
		fprintf(stderr, "%s [%lld-%lld] ", p, lb, ub);
		fprintf(stderr, "NUMBER NOT ACCEPTED %s\n", q);
		fail = 1;
	} else if (!ok && q == NULL) {
		fprintf(stderr, "%s [%lld-%lld] %lld ", p, lb, ub, val);
		fprintf(stderr, "NUMBER ACCEPTED\n");
		fail = 1;
	}
}

int main(void)
{
	test("1", 0, 10, 1);
	test("0", -2, 5, 1);
	test("0", 2, 5, 0);
	test("0", 2, LLONG_MAX, 0);
	test("-2", 0, LLONG_MAX, 0);
	test("0", -5, LLONG_MAX, 1);
	test("-3", -3, LLONG_MAX, 1);
	test("-9223372036854775808", LLONG_MIN, LLONG_MAX, 1);
	test("9223372036854775807", LLONG_MIN, LLONG_MAX, 1);
	test("-9223372036854775809", LLONG_MIN, LLONG_MAX, 0);
	test("9223372036854775808", LLONG_MIN, LLONG_MAX, 0);
	test("1000000000000000000000000", LLONG_MIN, LLONG_MAX, 0);
	test("-1000000000000000000000000", LLONG_MIN, LLONG_MAX, 0);
	test("-2", 10, -1, 0);
	test("-2", -10, -1, 1);
	test("-20", -10, -1, 0);
	test("20", -10, -1, 0);

	return (fail);
}

