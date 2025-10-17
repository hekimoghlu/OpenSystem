/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
/* OBSOLETE 19961031 -- for shared library compatibility */

#include	"sfhdr.h"

#undef	_sfgetl2

_BEGIN_EXTERNS_
#if _BLD_sfio && defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern long	_sfgetl2 _ARG_((Sfio_t*, long));

#undef	extern
_END_EXTERNS_

#if __STD_C
long _sfgetl2(reg Sfio_t* f, long v)
#else
long _sfgetl2(f, v)
reg Sfio_t*	f;
long		v;
#endif
{
	if (v < 0)
		return -1;
	sfungetc(f, v);
	return sfgetl(f);
}
