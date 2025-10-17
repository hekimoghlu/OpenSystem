/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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

#define	PASKHA		"paskha"
#define	PASKHALEN	(sizeof(PASKHA) - 1)

/* return difference in days between Julian and Gregorian calendars */
int
j2g(int year)
{
	return (year < 1500) ?
		0 :
		10 + (year/100 - 16) - ((year/100 - 16) / 4);
}

/* return year day for Orthodox Easter using Gauss formula */
/* (new style result) */

int
paskha(int R) /*year*/
{
	int a, b, c, d, e;
	static int x = 15;
	static int y = 6;
	int *cumday;

	a = R % 19;
	b = R % 4;
	c = R % 7;
	d = (19 * a + x) % 30;
	e = (2 * b + 4 * c + 6 * d + y) % 7;
	cumday = cumdaytab[isleap(R)];
	return (((cumday[3] + 1) + 22) + (d + e) + j2g(R));
}
