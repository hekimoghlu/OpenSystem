/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
 * AT&T Bell Laboratories
 *
 * return formatted <magicid.h> version string
 */

#include <ast.h>

char*
fmtversion(register unsigned long v)
{
	register char*	cur;
	register char*	end;
	char*		buf;
	int		n;

	buf = cur = fmtbuf(n = 18);
	end = cur + n;
	if (v >= 19700101L && v <= 29991231L)
		sfsprintf(cur, end - cur, "%04lu-%02lu-%02lu", (v / 10000) % 10000, (v / 100) % 100, v % 100);
	else
	{
		if (n = (v >> 24) & 0xff)
			cur += sfsprintf(cur, end - cur, "%d.", n);
		if (n = (v >> 16) & 0xff)
			cur += sfsprintf(cur, end - cur, "%d.", n);
		sfsprintf(cur, end - cur, "%ld.%ld", (v >> 8) & 0xff, v & 0xff);
	}
	return buf;
}
