/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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
 * AT&T Research
 *
 * sfio tmp string buffer support
 */

#include <sfio_t.h>
#include <ast.h>

#if __OBSOLETE__ >= 20070101 /* sfstr* macros now use sfsetbuf() */

NoN(sfstrtmp)

#else

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

/*
 * replace buffer in string stream f for either SF_READ or SF_WRITE
 */

extern int
sfstrtmp(register Sfio_t* f, int mode, void* buf, size_t siz)
{
	if (!(f->_flags & SF_STRING))
		return -1;
	if (f->_flags & SF_MALLOC)
		free(f->_data);
	f->_flags &= ~(SF_ERROR|SF_MALLOC);
	f->mode = mode;
	f->_next = f->_data = (unsigned char*)buf;
	f->_endw = f->_endr = f->_endb = f->_data + siz;
	f->_size = siz;
	return 0;
}

#endif
