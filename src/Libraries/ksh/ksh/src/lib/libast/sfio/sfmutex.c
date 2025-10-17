/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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

/*	Obtain/release exclusive use of a stream.
**
**	Written by Kiem-Phong Vo.
*/

/* the main locking/unlocking interface */
#if __STD_C
int sfmutex(Sfio_t* f, int type)
#else
int sfmutex(f, type)
Sfio_t*	f;
int	type;
#endif
{
#if !vt_threaded
	NOTUSED(f); NOTUSED(type);
	return 0;
#else

	SFONCE();

	if(!f)
		return -1;

	if(!f->mutex)
	{	if(f->bits&SF_PRIVATE)
			return 0;

		vtmtxlock(_Sfmutex);
		f->mutex = vtmtxopen(NIL(Vtmutex_t*), VT_INIT);
		vtmtxunlock(_Sfmutex);
		if(!f->mutex)
			return -1;
	}

	if(type == SFMTX_LOCK)
		return vtmtxlock(f->mutex);
	else if(type == SFMTX_TRYLOCK)
		return vtmtxtrylock(f->mutex);
	else if(type == SFMTX_UNLOCK)
		return vtmtxunlock(f->mutex);
	else if(type == SFMTX_CLRLOCK)
		return vtmtxclrlock(f->mutex);
	else	return -1;
#endif /*vt_threaded*/
}
