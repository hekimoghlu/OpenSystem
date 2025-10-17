/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
static char sccsid[] = "@(#)killpg.c	8.1 (Berkeley) 6/2/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/compat-43/killpg.c,v 1.5 2007/01/09 00:27:49 imp Exp $");

#include <sys/types.h>
#include <signal.h>
#include <errno.h>

int __kill(pid_t pid, int sig, int posix);

#if __DARWIN_UNIX03
#define	_PID1ERR	EPERM
#define	_POSIXKILL	1
#else	/* !__DARWIN_UNIX03 */
#define	_PID1ERR	ESRCH
#define	_POSIXKILL	0
#endif	/* !__DARWIN_UNIX03 */

/*
 * Backwards-compatible killpg().
 */
int
killpg(pid_t pgid, int sig)
{
	if (pgid == 1) {
		errno = _PID1ERR;
		return (-1);
	}
	return (__kill(-pgid, sig, _POSIXKILL));
}
