/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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

/*	Get the size of a stream.
**
**	Written by Kiem-Phong Vo.
*/
#if __STD_C
Sfoff_t sfsize(Sfio_t* f)
#else
Sfoff_t sfsize(f)
Sfio_t*	f;
#endif
{
	Sfdisc_t*	disc;
	reg int		mode;
	Sfoff_t		s;
	SFMTXDECL(f);

	SFMTXENTER(f, (Sfoff_t)(-1));

	if((mode = f->mode&SF_RDWR) != (int)f->mode && _sfmode(f,mode,0) < 0)
		SFMTXRETURN(f, (Sfoff_t)(-1));

	if(f->flags&SF_STRING)
	{	SFSTRSIZE(f);
		SFMTXRETURN(f, f->extent);
	}

	SFLOCK(f,0);

	s = f->here;

	if(f->extent >= 0)
	{	if(f->flags&(SF_SHARE|SF_APPENDWR))
		{	for(disc = f->disc; disc; disc = disc->disc)
				if(disc->seekf)
					break;
			if(!_sys_stat || disc)
			{	Sfoff_t	e;
				if((e = SFSK(f,0,SEEK_END,disc)) >= 0)
					f->extent = e;
				if(SFSK(f,f->here,SEEK_SET,disc) != f->here)
					f->here = SFSK(f,(Sfoff_t)0,SEEK_CUR,disc);
			}
#if _sys_stat
			else
			{	sfstat_t	st;
				if(sysfstatf(f->file,&st) < 0)
					f->extent = -1;
				else if((f->extent = st.st_size) < f->here)
					f->here = SFSK(f,(Sfoff_t)0,SEEK_CUR,disc);
			}
#endif
		}

		if((f->flags&(SF_SHARE|SF_PUBLIC)) == (SF_SHARE|SF_PUBLIC))
			f->here = SFSK(f,(Sfoff_t)0,SEEK_CUR,f->disc);
	}

	if(f->here != s && (f->mode&SF_READ) )
	{	/* buffered data is known to be invalid */
#ifdef MAP_TYPE
		if((f->bits&SF_MMAP) && f->data)
		{	SFMUNMAP(f,f->data,f->endb-f->data);
			f->data = NIL(uchar*);
		}
#endif
		f->next = f->endb = f->endr = f->endw = f->data;
	}

	if(f->here < 0)
		f->extent = -1;
	else if(f->extent < f->here)
		f->extent = f->here;

	if((s = f->extent) >= 0)
	{	if(f->flags&SF_APPENDWR)
			s += (f->next - f->data);
		else if(f->mode&SF_WRITE)
		{	s = f->here + (f->next - f->data);
			if(s < f->extent)
				s = f->extent;
		}
	}

	SFOPEN(f,0);
	SFMTXRETURN(f, s);
}
