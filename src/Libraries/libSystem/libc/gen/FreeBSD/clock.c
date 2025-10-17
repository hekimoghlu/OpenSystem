/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
static char sccsid[] = "@(#)clock.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/clock.c,v 1.4 2007/01/09 00:27:53 imp Exp $");

#include <sys/param.h>
#include <sys/time.h>
#include <sys/resource.h>

/*
 * Convert usec to clock ticks; could do (usec * CLOCKS_PER_SEC) / 1000000,
 * but this would overflow if we switch to nanosec.
 */
#define	CONVTCK(r)	((r).tv_sec * CLOCKS_PER_SEC \
			 + (r).tv_usec / (1000000 / CLOCKS_PER_SEC))

clock_t
clock()
{
	struct rusage ru;

	if (getrusage(RUSAGE_SELF, &ru))
		return ((clock_t) -1);
	return((clock_t)((CONVTCK(ru.ru_utime) + CONVTCK(ru.ru_stime))));
}
