/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#pragma prototyped
/*
 * Glenn Fowler
 * AT&T Research
 *
 * internal representation conversion support
 */

#include <ast.h>
#include <swap.h>

/*
 * get int_n from b according to op
 */

intmax_t
swapget(int op, const void* b, int n)
{
	register unsigned char*	p;
	register unsigned char*	d;
	intmax_t		v;
	unsigned char		tmp[sizeof(intmax_t)];

	if (n > sizeof(intmax_t))
		n = sizeof(intmax_t);
	if (op) swapmem(op, b, d = tmp, n);
	else d = (unsigned char*)b;
	p = d + n;
	v = 0;
	while (d < p)
	{
		v <<= CHAR_BIT;
		v |= *d++;
	}
	return v;
}
