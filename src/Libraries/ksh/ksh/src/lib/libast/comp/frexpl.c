/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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
 * frexpl/ldexpl implementation
 */

#include <ast.h>

#include "FEATURE/float"

#if _lib_frexpl && _lib_ldexpl

NoN(frexpl)

#else

#ifndef LDBL_MAX_EXP
#define LDBL_MAX_EXP	DBL_MAX_EXP
#endif

#if defined(_ast_fltmax_exp_index) && defined(_ast_fltmax_exp_shift)

#define INIT()		_ast_fltmax_exp_t _pow_
#define pow2(i)		(_pow_.f=1,_pow_.e[_ast_fltmax_exp_index]+=((i)<<_ast_fltmax_exp_shift),_pow_.f)

#else

static _ast_fltmax_t	pow2tab[LDBL_MAX_EXP + 1];

static int
init(void)
{
	register int		x;
	_ast_fltmax_t		g;

	g = 1;
	for (x = 0; x < elementsof(pow2tab); x++)
	{
		pow2tab[x] = g;
		g *= 2;
	}
	return 0;
}

#define INIT()		(pow2tab[0]?0:init())

#define pow2(i)		(pow2tab[i])

#endif

#if !_lib_frexpl

#undef	frexpl

extern _ast_fltmax_t
frexpl(_ast_fltmax_t f, int* p)
{
	register int		k;
	register int		x;
	_ast_fltmax_t		g;

	INIT();

	/*
	 * normalize
	 */

	x = k = LDBL_MAX_EXP / 2;
	if (f < 1)
	{
		g = 1.0L / f;
		for (;;)
		{
			k = (k + 1) / 2;
			if (g < pow2(x))
				x -= k;
			else if (k == 1 && g < pow2(x+1))
				break;
			else
				x += k;
		}
		if (g == pow2(x))
			x--;
		x = -x;
	}
	else if (f > 1)
	{
		for (;;)
		{
			k = (k + 1) / 2;
			if (f > pow2(x))
				x += k;
			else if (k == 1 && f > pow2(x-1))
				break;
			else
				x -= k;
		}
		if (f == pow2(x))
			x++;
	}
	else
		x = 1;
	*p = x;

	/*
	 * shift
	 */

	x = -x;
	if (x < 0)
		f /= pow2(-x);
	else if (x < LDBL_MAX_EXP)
		f *= pow2(x);
	else
		f = (f * pow2(LDBL_MAX_EXP - 1)) * pow2(x - (LDBL_MAX_EXP - 1));
	return f;
}

#endif

#if !_lib_ldexpl

#undef	ldexpl

extern _ast_fltmax_t
ldexpl(_ast_fltmax_t f, register int x)
{
	INIT();
	if (x < 0)
		f /= pow2(-x);
	else if (x < LDBL_MAX_EXP)
		f *= pow2(x);
	else
		f = (f * pow2(LDBL_MAX_EXP - 1)) * pow2(x - (LDBL_MAX_EXP - 1));
	return f;
}

#endif

#endif
