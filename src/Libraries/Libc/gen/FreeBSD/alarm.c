/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
static char sccsid[] = "@(#)alarm.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/alarm.c,v 1.3 2007/01/09 00:27:53 imp Exp $");

/*
 * Backwards compatible alarm.
 */
#include <sys/time.h>
#include <unistd.h>

unsigned int
alarm(unsigned int secs)
{
	struct itimerval it, oitv;
	struct itimerval *itp = &it;

	timerclear(&itp->it_interval);
	itp->it_value.tv_sec = secs;
	itp->it_value.tv_usec = 0;
	if (setitimer(ITIMER_REAL, itp, &oitv) < 0)
		return (-1);
	if (oitv.it_value.tv_usec)
		oitv.it_value.tv_sec++;
	return (oitv.it_value.tv_sec);
}
