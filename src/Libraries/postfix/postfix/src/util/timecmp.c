/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#include "timecmp.h"

/* timecmp - wrap-safe time_t comparison */

int     timecmp(time_t t1, time_t t2)
{
    time_t  delta = t1 - t2;

    if (delta == 0)
	return 0;

#define UNSIGNED(type) ( ((type)-1) > ((type)0) )

    /*
     * With a constant switch value, the compiler will emit only the code for
     * the correct case, so the signed/unsigned test happens at compile time.
     */
    switch (UNSIGNED(time_t) ? 0 : 1) {
    case 0:
	return ((2 * delta > delta) ? 1 : -1);
    case 1:
	return ((delta > (time_t) 0) ? 1 : -1);
    }
}

#ifdef TEST
#include <assert.h>

 /*
  * Bit banging!! There is no official constant that defines the INT_MAX
  * equivalent of the off_t type. Wietse came up with the following macro
  * that works as long as off_t is some two's complement number.
  * 
  * Note, however, that C99 permits signed integer representations other than
  * two's complement.
  */
#include <limits.h>
#define __MAXINT__(T) ((T) (((((T) 1) << ((sizeof(T) * CHAR_BIT) - 1)) ^ ((T) -1))))

int     main(void)
{
    time_t  now = time((time_t *) 0);

    /* Test that it works for normal times */
    assert(timecmp(now + 10, now) > 0);
    assert(timecmp(now, now) == 0);
    assert(timecmp(now - 10, now) < 0);

    /* Test that it works at a boundary time */
    if (UNSIGNED(time_t))
	now = (time_t) -1;
    else
	now = __MAXINT__(time_t);

    assert(timecmp(now + 10, now) > 0);
    assert(timecmp(now, now) == 0);
    assert(timecmp(now - 10, now) < 0);

    return (0);
}

#endif
