/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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
 * return pointer to formatted clock() tics t
 * return value length is at most 6
 */

#include <ast.h>
#include <tm.h>

char*
fmtclock(register Sfulong_t t)
{
	register int		u;
	char*			buf;
	int			z;

	static unsigned int	clk_tck;

	if (!clk_tck)
	{
#ifdef CLOCKS_PER_SEC
		clk_tck = CLOCKS_PER_SEC;
#else
		if (!(clk_tck = (unsigned int)strtoul(astconf("CLK_TCK", NiL, NiL), NiL, 10)))
			clk_tck = 60;
#endif
	}
	if (t == 0)
		return "0";
	if (t == ((Sfulong_t)~0))
		return "%";
	t = (t * 1000000) / clk_tck;
	if (t < 1000)
		u = 'u';
	else if ((t /= 1000) < 1000)
		u = 'm';
	else
		return fmtelapsed(t / 10, 100);
	buf = fmtbuf(z = 7);
	sfsprintf(buf, z, "%I*u%cs", sizeof(t), t, u);
	return buf;
}
