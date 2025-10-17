/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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

/*	Write out a long value in a portable format
**
**	Written by Kiem-Phong Vo.
*/

#if __STD_C
int _sfputl(Sfio_t* f, Sflong_t v)
#else
int _sfputl(f,v)
Sfio_t*		f;	/* write a portable long to this stream */
Sflong_t	v;	/* the value to be written */
#endif
{
#define N_ARRAY		(2*sizeof(Sflong_t))
	reg uchar	*s, *ps;
	reg ssize_t	n, p;
	uchar		c[N_ARRAY];
	SFMTXDECL(f);

	SFMTXENTER(f,-1);
	if(f->mode != SF_WRITE && _sfmode(f,SF_WRITE,0) < 0)
		SFMTXRETURN(f, -1);
	SFLOCK(f,0);

	s = ps = &(c[N_ARRAY-1]);
	if(v < 0)
	{	/* add 1 to avoid 2-complement problems with -SF_MAXINT */
		v = -(v+1);
		*s = (uchar)(SFSVALUE(v) | SF_SIGN);
	}
	else	*s = (uchar)(SFSVALUE(v));
	v = (Sfulong_t)v >> SF_SBITS;

	while(v > 0)
	{	*--s = (uchar)(SFUVALUE(v) | SF_MORE);
		v = (Sfulong_t)v >> SF_UBITS;
	}
	n = (ps-s)+1;

	if(n > 8 || SFWPEEK(f,ps,p) < n)
		n = SFWRITE(f,(Void_t*)s,n); /* write the hard way */
	else
	{	switch(n)
		{
		case 8 : *ps++ = *s++;
		case 7 : *ps++ = *s++;
		case 6 : *ps++ = *s++;
		case 5 : *ps++ = *s++;
		case 4 : *ps++ = *s++;
		case 3 : *ps++ = *s++;
		case 2 : *ps++ = *s++;
		case 1 : *ps++ = *s++;
		}
		f->next = ps;
	}

	SFOPEN(f,0);
	SFMTXRETURN(f, n);
}
