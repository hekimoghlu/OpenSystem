/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
 * return the swap operation for external to internal conversion
 * if size<0 then (-size) used and (-size==4)&&(op==3) => op=7
 * this is a workaround for 4 byte magic predicting 8 byte swap
 */

int
swapop(const void* internal, const void* external, int size)
{
	register int	op;
	register int	z;
	char		tmp[sizeof(intmax_t)];

	if ((z = size) < 0)
		z = -z;
	if (z <= 1)
		return 0;
	if (z <= sizeof(intmax_t))
		for (op = 0; op < z; op++)
			if (!memcmp(internal, swapmem(op, external, tmp, z), z))
			{
				if (size < 0 && z == 4 && op == 3)
					op = 7;
				return op;
			}
	return -1;
}
