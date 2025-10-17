/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "calendar.h"

/* return year day for Easter */

/*
 * This code is based on the Calendar FAQ's code for how to calculate
 * easter is. This is the Gregorian calendar version. They refer to
 * the Algorithm of Oudin in the "Explanatory Supplement to the
 * Astronomical Almanac".
 */

int
easter(int year) /* 0 ... abcd, NOT since 1900 */
{
	int G,	/* Golden number - 1 */
	    C,	/* Century */
	    H,	/* 23 - epact % 30 */
	    I,	/* days from 21 March to Paschal full moon */
	    J,	/* weekday of full moon */
	    L;	/* days from 21 March to Sunday on of before full moon */

	G = year % 19;
	C = year / 100;
	H = (C - C / 4 - (8 * C + 13) / 25 + 19 * G + 15) % 30;
	I = H - (H / 28) * (1 - (H / 28) * (29 / (H + 1)) * ((21 - G) / 11));
	J = (year + year / 4 + I + 2 - C + C / 4) % 7;

	L = I - J;

	if (isleap(year))
		return 31 + 29 + 21 + L + 7;
	else
		return 31 + 28 + 21 + L + 7;
}
