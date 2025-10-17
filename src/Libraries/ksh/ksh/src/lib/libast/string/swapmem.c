/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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
 * swap n bytes according to op
 * from==to is ok
 */

void*
swapmem(int op, const void* from, void* to, register size_t n)
{
	register char*	f = (char*)from;
	register char*	t = (char*)to;
	register int	c;

	switch (op & (n - 1))
	{
	case 0:
		if (t != f)
			memcpy(t, f, n);
		break;
	case 1:
		for (n >>= 1; n--; f += 2, t += 2)
		{
			c = f[0]; t[0] = f[1]; t[1] = c;
		}
		break;
	case 2:
		for (n >>= 2; n--; f += 4, t += 4)
		{
			c = f[0]; t[0] = f[2]; t[2] = c;
			c = f[1]; t[1] = f[3]; t[3] = c;
		}
		break;
	case 3:
		for (n >>= 2; n--; f += 4, t += 4)
		{
			c = f[0]; t[0] = f[3]; t[3] = c;
			c = f[1]; t[1] = f[2]; t[2] = c;
		}
		break;
	case 4:
		for (n >>= 3; n--; f += 8, t += 8)
		{
			c = f[0]; t[0] = f[4]; t[4] = c;
			c = f[1]; t[1] = f[5]; t[5] = c;
			c = f[2]; t[2] = f[6]; t[6] = c;
			c = f[3]; t[3] = f[7]; t[7] = c;
		}
		break;
	case 5:
		for (n >>= 3; n--; f += 8, t += 8)
		{
			c = f[0]; t[0] = f[5]; t[5] = c;
			c = f[1]; t[1] = f[4]; t[4] = c;
			c = f[2]; t[2] = f[7]; t[7] = c;
			c = f[3]; t[3] = f[6]; t[6] = c;
		}
		break;
	case 6:
		for (n >>= 3; n--; f += 8, t += 8)
		{
			c = f[0]; t[0] = f[6]; t[6] = c;
			c = f[1]; t[1] = f[7]; t[7] = c;
			c = f[2]; t[2] = f[4]; t[4] = c;
			c = f[3]; t[3] = f[5]; t[5] = c;
		}
		break;
	case 7:
		for (n >>= 3; n--; f += 8, t += 8)
		{
			c = f[0]; t[0] = f[7]; t[7] = c;
			c = f[1]; t[1] = f[6]; t[6] = c;
			c = f[2]; t[2] = f[5]; t[5] = c;
			c = f[3]; t[3] = f[4]; t[4] = c;
		}
		break;
	}
	return to;
}
