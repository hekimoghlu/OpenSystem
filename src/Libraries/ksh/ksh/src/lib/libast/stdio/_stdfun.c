/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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

#include <ast.h>

#if !_UWIN

void _STUB_stdfun(){}

#else

#include <ast_windows.h>
#include <uwin.h>
#include <dlfcn.h>
#include "FEATURE/uwin"

#if _lib___iob_func
#define IOB		((char*)__iob_func())
#elif _lib___p__iob
#define IOB		((char*)__p__iob())
#elif _dat__iob
#define IOB		((char*)_iob)
#else
#define IOB		((char*)_p__iob())
#endif

#define IOBMAX		(512*32)

#include "stdhdr.h"

int
_stdfun(Sfio_t* f, Funvec_t* vp)
{
	static char*	iob;
	static int	init;
	static void*	bp;
	static void*	np;

	if (!iob && !(iob = IOB))
		return 0;
	if (f && ((char*)f < iob || (char*)f > iob+IOBMAX))
		return 0;
	if (!vp->vec[1])
	{
		if (!init)
		{
			init = 1;
			bp = dlopen("/usr/bin/stdio.dll", 0);
		}
		if (bp && (vp->vec[1] = (Fun_f)dlsym(bp, vp->name)))
			return 1;
		if (!np && !(np = dlopen("/sys/msvcrt.dll", 0)))
			return -1;
		if (!(vp->vec[1] = (Fun_f)dlsym(np, vp->name)))
			return -1;
	}
	return 1;
}

#endif
