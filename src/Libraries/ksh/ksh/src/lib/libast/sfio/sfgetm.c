/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
#include	"sfhdr.h"

/*	Read an unsigned long value coded portably for a given range.
**
**	Written by Kiem-Phong Vo
*/

#if __STD_C
Sfulong_t sfgetm(Sfio_t* f, Sfulong_t m)
#else
Sfulong_t sfgetm(f, m)
Sfio_t*		f;
Sfulong_t	m;
#endif
{
	Sfulong_t	v;
	reg uchar	*s, *ends, c;
	reg int		p;
	SFMTXDECL(f);

	SFMTXENTER(f, (Sfulong_t)(-1));

	if(f->mode != SF_READ && _sfmode(f,SF_READ,0) < 0)
		SFMTXRETURN(f, (Sfulong_t)(-1));

	SFLOCK(f,0);

	for(v = 0;; )
	{	if(SFRPEEK(f,s,p) <= 0)
		{	f->flags |= SF_ERROR;
			v = (Sfulong_t)(-1);
			goto done;
		}
		for(ends = s+p; s < ends;)
		{	c = *s++;
			v = (v << SF_BBITS) | SFBVALUE(c);
			if((m >>= SF_BBITS) <= 0)
			{	f->next = s;
				goto done;
			}
		}
		f->next = s;
	}
done:
	SFOPEN(f,0);
	SFMTXRETURN(f, v);
}
