/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
static char sccsid[] = "@(#)times.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/times.c,v 1.2 2002/02/01 01:08:48 obrien Exp $");

#include <sys/param.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/resource.h>

/*
 * Convert usec to clock ticks; could do (usec * CLK_TCK) / 1000000,
 * but this would overflow if we switch to nanosec.
 */
#define	CONVTCK(r)	(r.tv_sec * CLK_TCK + r.tv_usec / (1000000 / CLK_TCK))

clock_t
times(tp)
	struct tms *tp;
{
	struct rusage ru;
	struct timeval t;

	if (getrusage(RUSAGE_SELF, &ru) < 0)
		return ((clock_t)-1);
	tp->tms_utime = CONVTCK(ru.ru_utime);
	tp->tms_stime = CONVTCK(ru.ru_stime);
	if (getrusage(RUSAGE_CHILDREN, &ru) < 0)
		return ((clock_t)-1);
	tp->tms_cutime = CONVTCK(ru.ru_utime);
	tp->tms_cstime = CONVTCK(ru.ru_stime);
	if (gettimeofday(&t, (struct timezone *)0))
		return ((clock_t)-1);
	return ((clock_t)(CONVTCK(t)));
}
