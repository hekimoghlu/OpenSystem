/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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
static char sccsid[] = "@(#)assert.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/assert.c,v 1.8 2007/01/09 00:27:53 imp Exp $");

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#if __has_include(<CrashReporterClient.h>)
#include <CrashReporterClient.h>
#else
#define CRGetCrashLogMessage() NULL
#define CRSetCrashLogMessage(...)
#endif
#include "_simple.h"

void
__assert_rtn(const char *func, const char *file, int line,
    const char *failedexpr)
{
	if (func == (const char *)-1L) {
		/* 8462256: special case to replace __eprintf */
		_simple_dprintf(STDERR_FILENO,
		     "%s:%d: failed assertion `%s'\n", file, line, failedexpr);
		if (!CRGetCrashLogMessage()) {
			_SIMPLE_STRING s = _simple_salloc();
			if (s) {
				_simple_sprintf(s,
				  "%s:%d: failed assertion `%s'\n",
				  file, line, failedexpr);
				CRSetCrashLogMessage(_simple_string(s));
			} else
				CRSetCrashLogMessage(failedexpr);
		}
	} else if (func == NULL) {
		_simple_dprintf(STDERR_FILENO,
		     "Assertion failed: (%s), file %s, line %d.\n", failedexpr,
		     file, line);
		if (!CRGetCrashLogMessage()) {
			_SIMPLE_STRING s = _simple_salloc();
			if (s) {
				_simple_sprintf(s,
				  "Assertion failed: (%s), file %s, line %d.\n",
				  failedexpr, file, line);
				CRSetCrashLogMessage(_simple_string(s));
			} else
				CRSetCrashLogMessage(failedexpr);
		}
	} else {
		_simple_dprintf(STDERR_FILENO,
		     "Assertion failed: (%s), function %s, file %s, line %d.\n",
		     failedexpr, func, file, line);
		if (!CRGetCrashLogMessage()) {
			_SIMPLE_STRING s = _simple_salloc();
			if (s) {
				_simple_sprintf(s,
				  "Assertion failed: (%s), function %s, file %s, line %d.\n",
				  failedexpr, func, file, line);
				CRSetCrashLogMessage(_simple_string(s));
			} else
				CRSetCrashLogMessage(failedexpr);
		}
	}
	abort();
	/* NOTREACHED */
}
