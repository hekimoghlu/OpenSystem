/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
static char sccsid[] = "@(#)waitpid.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/waitpid.c,v 1.7 2007/01/09 00:27:56 imp Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include "un-namespace.h"

#if __DARWIN_UNIX03
#include <errno.h>
#endif /* __DARWIN_UNIX03 */
#ifdef VARIANT_CANCELABLE
int __wait4(pid_t, int *, int , struct rusage *);
#else /* !VARIANT_CANCELABLE */
int __wait4_nocancel(pid_t, int *, int , struct rusage *);
#endif /* VARIANT_CANCELABLE */

pid_t
__waitpid(pid_t pid, int *istat, int options)
{
#if __DARWIN_UNIX03
	/* POSIX: Validate waitpid() options before calling wait4() */
	if ((options & (WCONTINUED | WNOHANG | WUNTRACED)) != options) {
		errno = EINVAL;
		return ((pid_t)-1);
	}
#endif	/* __DARWIN_UNIX03 */

#ifdef VARIANT_CANCELABLE
	return (__wait4(pid, istat, options, (struct rusage *)0));
#else /* !VARIANT_CANCELABLE */
	return (__wait4_nocancel(pid, istat, options, (struct rusage *)0));
#endif /* VARIANT_CANCELABLE */
}

__weak_reference(__waitpid, waitpid);
__weak_reference(__waitpid, _waitpid);
