/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunreachable-code"

#include <sys/cdefs.h>
#ifndef lint
#ifndef NOID
static char	elsieid[] __unused = "@(#)difftime.c	8.1";
#endif /* !defined NOID */
#endif /* !defined lint */
__FBSDID("$FreeBSD: src/lib/libc/stdtime/difftime.c,v 1.9 2009/05/23 06:31:50 edwin Exp $");

/*LINTLIBRARY*/

#include "namespace.h"
#include "private.h"	/* for time_t, TYPE_INTEGRAL, and TYPE_SIGNED */
#include "un-namespace.h"

double
difftime(time1, time0)
const time_t	time1;
const time_t	time0;
{
	/*
	** If (sizeof (double) > sizeof (time_t)) simply convert and subtract
	** (assuming that the larger type has more precision).
	** This is the common real-world case circa 2004.
	*/
	if (sizeof (double) > sizeof (time_t))
		return (double) time1 - (double) time0;
	if (!TYPE_INTEGRAL(time_t)) {
		/*
		** time_t is floating.
		*/
		return time1 - time0;
	}
	if (!TYPE_SIGNED(time_t)) {
		/*
		** time_t is integral and unsigned.
		** The difference of two unsigned values can't overflow
		** if the minuend is greater than or equal to the subtrahend.
		*/
		if (time1 >= time0)
			return time1 - time0;
		else	return -((double) (time0 - time1));
	}
	/*
	** time_t is integral and signed.
	** Handle cases where both time1 and time0 have the same sign
	** (meaning that their difference cannot overflow).
	*/
	if ((time1 < 0) == (time0 < 0))
		return time1 - time0;
	/*
	** time1 and time0 have opposite signs.
	** Punt if unsigned long is too narrow.
	*/
	if (sizeof (unsigned long) < sizeof (time_t))
		return (double) time1 - (double) time0;
	/*
	** Stay calm...decent optimizers will eliminate the complexity below.
	*/
	if (time1 >= 0 /* && time0 < 0 */)
		return (unsigned long) time1 +
			(unsigned long) (-(time0 + 1)) + 1;
	return -(double) ((unsigned long) time0 +
		(unsigned long) (-(time1 + 1)) + 1);
}
#pragma clang diagnostic pop
