/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
** This file is in the public domain, so clarified as of
** 1996-06-05 by Arthur David Olson.
*/

/*LINTLIBRARY*/

#include "private.h"	/* for time_t and TYPE_SIGNED */

/* Return -X as a double.  Using this avoids casting to 'double'.  */
static double
dminus(double x)
{
  return -x;
}

double
difftime(time_t time1, time_t time0)
{
	/*
	** If double is large enough, simply convert and subtract
	** (assuming that the larger type has more precision).
	*/
	if (sizeof(time_t) < sizeof(double)) {
	  double t1 = time1, t0 = time0;
	  return t1 - t0;
	}

	/*
	** The difference of two unsigned values can't overflow
	** if the minuend is greater than or equal to the subtrahend.
	*/
	if (!TYPE_SIGNED(time_t))
	  return time0 <= time1 ? time1 - time0 : dminus(time0 - time1);

	/* Use uintmax_t if wide enough.  */
	if (sizeof(time_t) <= sizeof(uintmax_t)) {
	  uintmax_t t1 = time1, t0 = time0;
	  return time0 <= time1 ? t1 - t0 : dminus(t0 - t1);
	}

	/*
	** Handle cases where both time1 and time0 have the same sign
	** (meaning that their difference cannot overflow).
	*/
	if ((time1 < 0) == (time0 < 0))
	  return time1 - time0;

	/*
	** The values have opposite signs and uintmax_t is too narrow.
	** This suffers from double rounding; attempt to lessen that
	** by using long double temporaries.
	*/
	{
	  long double t1 = time1, t0 = time0;
	  return t1 - t0;
	}
}
