/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

/*	Push back one byte to a given SF_READ stream
**
**	Written by Kiem-Phong Vo.
*/
#if __STD_C
static int _uexcept(Sfio_t* f, int type, Void_t* val, Sfdisc_t* disc)
#else
static int _uexcept(f,type,val,disc)
Sfio_t		*f;
int		type;
Void_t*		val;
Sfdisc_t	*disc;
#endif
{	
	NOTUSED(val);

	/* hmm! This should never happen */
	if(disc != _Sfudisc)
		return -1;

	/* close the unget stream */
	if(type != SF_CLOSING)
		(void)sfclose((*_Sfstack)(f,NIL(Sfio_t*)));

	return 1;
}

#if __STD_C
int sfungetc(Sfio_t* f, int c)
#else
int sfungetc(f,c)
Sfio_t*		f;	/* push back one byte to this stream */
int		c;	/* the value to be pushed back */
#endif
{
	reg Sfio_t*	uf;
	SFMTXDECL(f);

	SFMTXENTER(f, -1);

	if(c < 0 || (f->mode != SF_READ && _sfmode(f,SF_READ,0) < 0))
		SFMTXRETURN(f, -1);
	SFLOCK(f,0);

	/* fast handling of the typical unget */
	if(f->next > f->data && f->next[-1] == (uchar)c)
	{	f->next -= 1;
		goto done;
	}

	/* make a string stream for unget characters */
	if(f->disc != _Sfudisc)
	{	if(!(uf = sfnew(NIL(Sfio_t*),NIL(char*),(size_t)SF_UNBOUND,
				-1,SF_STRING|SF_READ)))
		{	c = -1;
			goto done;
		}
		_Sfudisc->exceptf = _uexcept;
		sfdisc(uf,_Sfudisc);
		SFOPEN(f,0); (void)sfstack(f,uf); SFLOCK(f,0);
	}

	/* space for data */
	if(f->next == f->data)
	{	reg uchar*	data;
		if(f->size < 0)
			f->size = 0;
		if(!(data = (uchar*)malloc(f->size+16)))
		{	c = -1;
			goto done;
		}
		f->flags |= SF_MALLOC;
		if(f->data)
			memcpy((char*)(data+16),(char*)f->data,f->size);
		f->size += 16;
		f->data  = data;
		f->next  = data+16;
		f->endb  = data+f->size;
	}

	*--f->next = (uchar)c;
done:
	SFOPEN(f,0);
	SFMTXRETURN(f, c);
}
