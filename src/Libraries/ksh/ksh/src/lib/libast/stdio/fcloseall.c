/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

#include "stdhdr.h"

#define MAXLOOP		3

int
fcloseall(void)
{
	Sfpool_t*	p;
	Sfpool_t*	next;
	int		n;
	int		nclose;
	int		count;
	int		loop;

	STDIO_INT(0, "fcloseall", int, (void), ())

	for(loop = 0; loop < MAXLOOP; ++loop)
	{	nclose = count = 0;
		for(p = &_Sfpool; p; p = next)
		{	/* find the next legitimate pool */
			for(next = p->next; next; next = next->next)
				if(next->n_sf > 0)
					break;
			for(n = 0; n < ((p == &_Sfpool) ? p->n_sf : 1); ++n)
			{	count += 1;
				if(sfclose(p->sf[n]) >= 0)
					nclose += 1;
			}
		}
		if(nclose == count)
			break;
	}
	return 0; /* always return 0 per GNU */
}
