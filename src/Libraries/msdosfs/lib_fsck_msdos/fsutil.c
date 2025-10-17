/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
 * Copyright (c) 1990, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <sys/cdefs.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#if __STDC__
#include <stdarg.h>
#else
#include <varargs.h>
#endif
#include <errno.h>
#include <fstab.h>
#include <err.h>

#include <sys/types.h>
#include <sys/stat.h>

#include "ext.h"
#include "fsutil.h"

extern char *__progname;

static void vmsg __P((int, const char *, va_list));

/*VARARGS*/
void
#if __STDC__
errexit(const char *fmt, ...)
#else
errexit(va_alist)
	va_dcl
#endif
{
	va_list ap;

#if __STDC__
	va_start(ap, fmt);
#else
	const char *fmt;

	va_start(ap);
	fmt = va_arg(ap, const char *);
#endif
	(void) vfprintf(stderr, fmt, ap);
	va_end(ap);
	exit(8);
}

static void
vmsg(int fatal, const char *fmt, va_list ap)
{
	if (!fatal && fsck_preen())
		(void) printf("%s: ", fsck_dev());
	
	if (!fsck_quiet())
		(void) vprintf(fmt, ap);

	if (fatal && fsck_preen())
		(void) printf("\n");

	if (fatal && fsck_preen()) {
		(void) printf("%s: UNEXPECTED INCONSISTENCY; RUN %s MANUALLY.\n", fsck_dev(), __progname);
		exit(8);
	}
}

/*VARARGS*/
void
#if __STDC__
pfatal(const char *fmt, ...)
#else
pfatal(va_alist)
	va_dcl
#endif
{
	va_list ap;

#if __STDC__
	va_start(ap, fmt);
#else
	const char *fmt;

	va_start(ap);
	fmt = va_arg(ap, const char *);
#endif
	vpfatal(NULL, fmt, ap);
	va_end(ap);
}

/*VARARGS*/
void
#if __STDC__
pwarn(const char *fmt, ...)
#else
pwarn(va_alist)
	va_dcl
#endif
{
	va_list ap;
#if __STDC__
	va_start(ap, fmt);
#else
	const char *fmt;

	va_start(ap);
	fmt = va_arg(ap, const char *);
#endif
	vpwarn(NULL, fmt, ap);
	va_end(ap);
}

#ifdef __APPLE__
__private_extern__
#endif
void
perrno(const char *s)
{
	pfatal("%s (%s)\n", s, strerror(errno));
}

void perr(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vperr(NULL, fmt, ap);
	va_end(ap);
}

void vpfatal(fsck_client_ctx_t client, const char *fmt, va_list ap)
{
	vmsg(1, fmt, ap);
}

void vpwarn(fsck_client_ctx_t client, const char *fmt, va_list ap)
{
	vmsg(0, fmt, ap);
}

void vperr(fsck_client_ctx_t client, const char *fmt, va_list ap)
{
	if (fsck_preen()) {
		(void) fprintf(stderr, "%s: ", fsck_dev());
	}
	if (!fsck_quiet()) {
		(void) vfprintf(stderr, fmt, ap);
	}
}

void vprint(fsck_client_ctx_t client, int level, const char *fmt, va_list ap)
{
    switch (level) {
        case LOG_INFO:
            vprintf(fmt, ap);
            break;
        case LOG_ERR:
            vperr(client, fmt, ap);
            break;
        case LOG_CRIT:
            vpfatal(client, fmt, ap);
            break;
        default:
            break;
    }
}

void fsck_print(lib_fsck_ctx_t c, int level, const char *fmt, ...)
{
	if (c.print) {
		va_list ap;
		va_start(ap, fmt);
		c.print(c.client_ctx, level, fmt, ap);
		va_end(ap);
	}
}

int fsck_ask(lib_fsck_ctx_t c, int def, const char *fmt, ...)
{
	if (c.ask) {
		va_list ap;
		va_start(ap, fmt);
		int retval = c.ask(c.client_ctx, def, fmt, ap);
		va_end(ap);
		return retval;
	}
	return -1;
}
