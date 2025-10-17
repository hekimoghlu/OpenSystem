/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#include "lafe_platform.h"
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "err.h"

static void lafe_vwarnc(int, const char *, va_list) __LA_PRINTFLIKE(2, 0);

static const char *lafe_progname;

const char *
lafe_getprogname(void)
{

	return lafe_progname;
}

void
lafe_setprogname(const char *name, const char *defaultname)
{

	if (name == NULL)
		name = defaultname;
#if defined(_WIN32) && !defined(__CYGWIN__)
	lafe_progname = strrchr(name, '\\');
	if (strrchr(name, '/') > lafe_progname)
#endif
	lafe_progname = strrchr(name, '/');
	if (lafe_progname != NULL)
		lafe_progname++;
	else
		lafe_progname = name;
}

static void
lafe_vwarnc(int code, const char *fmt, va_list ap)
{
	fprintf(stderr, "%s: ", lafe_progname);
	vfprintf(stderr, fmt, ap);
	if (code != 0)
		fprintf(stderr, ": %s", strerror(code));
	fprintf(stderr, "\n");
}

void
lafe_warnc(int code, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	lafe_vwarnc(code, fmt, ap);
	va_end(ap);
}

void
lafe_errc(int eval, int code, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	lafe_vwarnc(code, fmt, ap);
	va_end(ap);
	exit(eval);
}
