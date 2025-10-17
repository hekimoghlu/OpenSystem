/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#if defined(LIBC_SCCS) && !defined(lint)
__SCCSID("@(#)ualarm.c	8.1 (Berkeley) 6/4/93");
__FBSDID("$FreeBSD$");
#endif /* LIBC_SCCS and not lint */

#include <sys/time.h>
#include <unistd.h>

#define	USPS	1000000		/* # of microseconds in a second */

/*
 * Generate a SIGALRM signal in ``usecs'' microseconds.
 * If ``reload'' is non-zero, keep generating SIGALRM
 * every ``reload'' microseconds after the first signal.
 */
useconds_t
ualarm(useconds_t usecs, useconds_t reload)
{
	struct itimerval new, old;

	new.it_interval.tv_usec = reload % USPS;
	new.it_interval.tv_sec = reload / USPS;

	new.it_value.tv_usec = usecs % USPS;
	new.it_value.tv_sec = usecs / USPS;

	if (setitimer(ITIMER_REAL, &new, &old) == 0)
		return (old.it_value.tv_sec * USPS + old.it_value.tv_usec);
	return (-1);
}
