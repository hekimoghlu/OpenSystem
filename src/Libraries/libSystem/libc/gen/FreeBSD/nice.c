/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
static char sccsid[] = "@(#)nice.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/nice.c,v 1.4 2007/01/09 00:27:54 imp Exp $");

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <errno.h>
#include <unistd.h>
#if __DARWIN_UNIX03
#include <limits.h>
#endif /* __DARWIN_UNIX03 */
/*
 * Backwards compatible nice.
 */
int
nice(incr)
	int incr;
{
	int prio, rv;

	errno = 0;
	prio = getpriority(PRIO_PROCESS, 0);
	if (prio == -1 && errno)
		return (-1);
#if __DARWIN_UNIX03
	if (prio + incr > NZERO-1)
		incr = NZERO-1-prio;
#endif /* __DARWIN_UNIX03 */
	rv = setpriority(PRIO_PROCESS, 0, prio + incr);
	if (rv == -1 && errno == EACCES)
		errno = EPERM;
	return (rv == -1) ? rv : getpriority(PRIO_PROCESS, 0);
}
