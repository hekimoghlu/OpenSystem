/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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

/*	Read a portably coded double value
**
**	Written by Kiem-Phong Vo
*/

#if __STD_C
Sfdouble_t sfgetd(Sfio_t* f)
#else
Sfdouble_t sfgetd(f)
Sfio_t*	f;
#endif
{
	reg uchar	*s, *ends, c;
	reg int		p, sign, exp;
	Sfdouble_t	v;
	SFMTXDECL(f); /* declare a local stream variable for multithreading */

	SFMTXENTER(f,-1.);

	if((sign = sfgetc(f)) < 0 || (exp = (int)sfgetu(f)) < 0)
		SFMTXRETURN(f, -1.);

	if(f->mode != SF_READ && _sfmode(f,SF_READ,0) < 0)
		SFMTXRETURN(f, -1.);

	SFLOCK(f,0);

	v = 0.;
	for(;;)
	{	/* fast read for data */
		if(SFRPEEK(f,s,p) <= 0)
		{	f->flags |= SF_ERROR;
			v = -1.;
			goto done;
		}

		for(ends = s+p; s < ends; )
		{	c = *s++;
			v += SFUVALUE(c);
			v = ldexpl(v,-SF_PRECIS);
			if(!(c&SF_MORE))
			{	f->next = s;
				goto done;
			}
		}
		f->next = s;
	}

done:
	v = ldexpl(v,(sign&02) ? -exp : exp);
	if(sign&01)
		v = -v;

	SFOPEN(f,0);
	SFMTXRETURN(f, v);
}
