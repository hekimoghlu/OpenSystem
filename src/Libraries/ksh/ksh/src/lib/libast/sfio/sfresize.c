/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

/*	Resize a stream.
	Written by Kiem-Phong Vo.
*/

#if __STD_C
int sfresize(Sfio_t* f, Sfoff_t size)
#else
int sfresize(f, size)
Sfio_t*		f;
Sfoff_t		size;
#endif
{
	SFMTXDECL(f);

	SFMTXENTER(f, -1);

	if(size < 0 || f->extent < 0 ||
	   (f->mode != SF_WRITE && _sfmode(f,SF_WRITE,0) < 0) )
		SFMTXRETURN(f, -1);

	SFLOCK(f,0);

	if(f->flags&SF_STRING)
	{	SFSTRSIZE(f);

		if(f->extent >= size)
		{	if((f->flags&SF_MALLOC) && (f->next - f->data) <= size)
			{	size_t	s = (((size_t)size + 1023)/1024)*1024;
				Void_t*	d;
				if(s < f->size && (d = realloc(f->data, s)) )
				{	f->data = d;
					f->size = s;
					f->extent = s;
				}
			}
			memclear((char*)(f->data+size), (int)(f->extent-size));
		}
		else
		{	if(SFSK(f, size, SEEK_SET, f->disc) != size)
				SFMTXRETURN(f, -1);
			memclear((char*)(f->data+f->extent), (int)(size-f->extent));
		}
	}
	else
	{	if(f->next > f->data)
			SFSYNC(f);
#if _lib_ftruncate
		if(ftruncate(f->file, (sfoff_t)size) < 0)
			SFMTXRETURN(f, -1);
#else
		SFMTXRETURN(f, -1);
#endif
	}

	f->extent = size;

	SFOPEN(f, 0);

	SFMTXRETURN(f, 0);
}
